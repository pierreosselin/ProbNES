from typing import Optional, Any, Union, Tuple
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf

from botorch.models.model import Model
from torch import Tensor
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
import torch
from gpytorch.utils.cholesky import psd_safe_cholesky
import botorch
from module.cholesky import one_step_cholesky
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.utils.transforms import unnormalize
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior


class GradientInformation(botorch.acquisition.AnalyticAcquisitionFunction):
    """Acquisition function to sample points for gradient information.

    Attributes:
        model: Gaussian process model that supplies the Jacobian (e.g. DerivativeExactGPSEModel).
    """

    def __init__(self, model):
        """Inits acquisition function with model."""
        super().__init__(model)

    def update_theta_i(self, theta_i: torch.Tensor):
        """Updates the current parameters.

        This leads to an update of K_xX_dx.

        Args:
            theta_i: New parameters.
        """
        if not torch.is_tensor(theta_i):
            theta_i = torch.tensor(theta_i)
        self.theta_i = theta_i
        self.update_K_xX_dx()

    def update_K_xX_dx(self):
        """When new x is given update K_xX_dx."""
        # Pre-compute large part of K_xX_dx.
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, self.model.D)
        self.K_xX_dx_part = self._get_KxX_dx(x, X)

    def _get_KxX_dx(self, x, X) -> torch.Tensor:
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        N = X.shape[0]
        n = x.shape[0]
        K_xX = self.model.covar_module(x, X).evaluate()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.model.D, device=X.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.model.D) - X.view(1, N, self.model.D))
                * K_xX.view(n, N, 1)
            ).transpose(1, 2)
        )

    # TODO: nicer batch-update for batch of thetas.
    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (b) x D-dim Tensor of (b) batches with a d-dim theta points each.

        Returns:
            A (b)-dim Tensor of acquisition function values at the given theta points.
        """
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, D)

        variances = []
        for theta in thetas:
            theta = theta.view(-1, D)
            # Compute K_Xθ, K_θθ (do not forget to add noise).
            K_Xθ = self.model.covar_module(X, theta).evaluate()
            K_θθ = self.model.covar_module(theta).evaluate() + sigma_n * torch.eye(
                K_Xθ.shape[-1]
            ).to(theta)

            # Get Cholesky factor.
            L = one_step_cholesky(
                top_left=self.model.get_L_lower().transpose(-1, -2),
                K_Xθ=K_Xθ,
                K_θθ=K_θθ,
                A_inv=self.model.get_KXX_inv(),
            )

            # Get K_XX_inv.
            K_XX_inv = torch.cholesky_inverse(L, upper=True)

            # get K_xX_dx
            K_xθ_dx = self._get_KxX_dx(x, theta)
            K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

            # Compute_variance.
            variance_d = -K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(1, 2)
            variances.append(torch.trace(variance_d.view(D, D)).view(1))

        return -torch.cat(variances, dim=0)


class DownhillQuadratic(GradientInformation):
    def update_theta_i(self, theta_i: torch.Tensor):
        super().update_theta_i(theta_i)

        self.mean_d, _ = self.model.posterior_derivative(self.theta_i)

    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward_old(self, x):
        sigma_f = self.model.covar_module.outputscale.detach()
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]

        x = x.view(-1, D)

        # Compute K_Xθ, K_θθ (do not forget to add noise).
        K_Xθ = self.model.covar_module(X, x).evaluate()
        K_θθ = self.model.covar_module(x).evaluate() + sigma_n * torch.eye(
            K_Xθ.shape[-1]
        ).to(x)

        # Get Cholesky factor.
        L = one_step_cholesky(
            top_left=self.model.get_L_lower().transpose(-1, -2),
            K_Xθ=K_Xθ,
            K_θθ=K_θθ,
            A_inv=self.model.get_KXX_inv(),
        )

        # Get K_XX_inv.
        K_XX_inv = torch.cholesky_inverse(L, upper=True)

        # get K_xX_dx
        K_xθ_dx = self._get_KxX_dx(self.theta_i.view(-1, D), x)
        K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

        # Compute_variance.
        covar_xstar_xstar_condx = (
            self.model._get_Kxx_dx2() - K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(-1, -2)
        )

        L_xstar_xstar_condx = psd_safe_cholesky(covar_xstar_xstar_condx)
        covar_xstar_x = K_xθ_dx - self.K_xX_dx_part @ self.model.get_KXX_inv() @ K_Xθ
        covar_x_x = K_θθ - K_Xθ.transpose(-1, -2) @ self.model.get_KXX_inv() @ K_Xθ

        Lxx = psd_safe_cholesky(covar_x_x)
        A = torch.triangular_solve(
            covar_xstar_x.transpose(-1, -2), Lxx, upper=False
        ).solution.transpose(-1, -2)

        LinvMu = torch.triangular_solve(
            self.mean_d.unsqueeze(-1), L_xstar_xstar_condx, upper=False
        ).solution
        LinvA = torch.triangular_solve(A, L_xstar_xstar_condx, upper=False).solution

        return (LinvMu.square().sum() + LinvA.square().sum()).unsqueeze(0)

    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, x):
        sigma_f = self.model.covar_module.outputscale.detach()
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]

        results = []

        for x_i in x:
            x_i = x_i.view(-1, D)

            # Compute K_Xθ, K_θθ (do not forget to add noise).
            K_Xθ = self.model.covar_module(X, x_i).evaluate()
            K_θθ = self.model.covar_module(x_i).evaluate() + sigma_n * torch.eye(
                K_Xθ.shape[-1]
            ).to(x_i)

            # Get Cholesky factor.
            L = one_step_cholesky(
                top_left=self.model.get_L_lower().transpose(-1, -2),
                K_Xθ=K_Xθ,
                K_θθ=K_θθ,
                A_inv=self.model.get_KXX_inv(),
            )

            # Get K_XX_inv.
            K_XX_inv = torch.cholesky_inverse(L, upper=True)

            # get K_xX_dx
            K_xθ_dx = self._get_KxX_dx(self.theta_i.view(-1, D), x_i)
            K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

            # Compute_variance.
            covar_xstar_xstar_condx = (
                self.model._get_Kxx_dx2()
                - K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(-1, -2)
            )

            try:
                L_xstar_xstar_condx = psd_safe_cholesky(
                    covar_xstar_xstar_condx, max_tries=9
                )
            except:
                from IPython.core.debugger import set_trace

                set_trace()
            covar_xstar_x = (
                K_xθ_dx - self.K_xX_dx_part @ self.model.get_KXX_inv() @ K_Xθ
            )
            covar_x_x = K_θθ - K_Xθ.transpose(-1, -2) @ self.model.get_KXX_inv() @ K_Xθ

            # from Kaiwen
            Lxx = psd_safe_cholesky(covar_x_x)
            A = torch.triangular_solve(
                covar_xstar_x.transpose(-1, -2), Lxx, upper=False
            ).solution.transpose(-1, -2)

            LinvMu = torch.triangular_solve(
                self.mean_d.unsqueeze(-1), L_xstar_xstar_condx, upper=False
            ).solution
            LinvA = torch.triangular_solve(A, L_xstar_xstar_condx, upper=False).solution

            results.append((LinvMu.square().sum() + LinvA.square().sum()).unsqueeze(0))

        results = torch.cat(results)

        return results

    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward_test(self, x):
        sigma_f = self.model.covar_module.outputscale.detach()
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]

        results = []

        for x_i in x:
            x_i = x_i.view(-1, D)

            L_tt = self.model.get_L_lower().transpose(-1, -2)
            K_tz = self.model.covar_module(X, x_i).evaluate()
            L_zt = torch.triangular_solve(K_tz, L_tt, upper=True).solution.T

            K_zz = self.model.covar_module(x_i)
            K_zz = K_zz.add_jitter(self.model.likelihood.noise).evaluate()

            L_zz = psd_safe_cholesky(K_zz - L_zt.mm(L_zt.T))

            zero = L_tt.new_zeros((L_tt.size(0), L_zz.size(1)))
            L = torch.cat(
                (torch.cat((L_tt, zero), dim=1), torch.cat((L_zt, L_zz), dim=1)), dim=0
            )

            K_xx = self.model._get_Kxx_dx2()

            K_xt = self.K_xX_dx_part
            K_xz = self._get_KxX_dx(self.theta_i.view(-1, D), x_i)
            K_x_tz = torch.cat((K_xt, K_xz), dim=-1)

            covar_xstar_xstar_condx = K_xx - torch.matmul(
                K_x_tz, torch.cholesky_solve(K_x_tz.mT, L)
            )
            L_xstar_xstar_condx = psd_safe_cholesky(covar_xstar_xstar_condx)

            covar_xstar_x = K_xz - self.K_xX_dx_part @ self.model.get_KXX_inv() @ K_tz
            covar_x_x = K_zz - K_tz.transpose(-1, -2) @ self.model.get_KXX_inv() @ K_tz

            # from Kaiwen
            Lxx = psd_safe_cholesky(covar_x_x)
            A = torch.triangular_solve(
                covar_xstar_x.transpose(-1, -2), Lxx, upper=False
            ).solution.transpose(-1, -2)

            LinvMu = torch.triangular_solve(
                self.mean_d.unsqueeze(-1), L_xstar_xstar_condx, upper=False
            ).solution
            LinvA = torch.triangular_solve(A, L_xstar_xstar_condx, upper=False).solution

            results.append((LinvMu.square().sum() + LinvA.square().sum()).unsqueeze(0))

        results = torch.cat(results)

        return results


class piqExpectedImprovement(qExpectedImprovement):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        pi_distrib: Distribution,
        n_iter: int,
        beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.pi_dist = pi_distrib
        self.n_iter = n_iter
        self.beta = beta

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        r"""Evaluate qExpectedImprovement on the candidate set `X` with pi-BO weighting

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """

        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        pi_X = (self.beta / self.n_iter)*self.pi_dist.log_prob(X).sum(axis = 1)
        return q_ei * torch.exp(pi_X)
    


def optimize_acqf_custom_bo(
    acq_func: botorch.acquisition.AcquisitionFunction,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to optimize the GradientInformation acquisition function for custom Bayesian optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A 2 x D tensor of lower and upper bounds for each column of X.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition function optimization.
        raw_samples: The number of samples for initialization.

    Returns:
        A two-element tuple containing:
            - a q x D-dim tensor of generated candidates.
            - a tensor of associated acquisition values.
    """
    candidates, acq_value = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,  # Analytic acquisition function.
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # Used for initialization heuristic.
        options={"nonnegative": True, "batch_limit": 5},
        return_best_only=True,
        sequential=False,
    )
    # Observe new values.
    new_x = candidates.detach()
    return new_x, acq_value


def optimize_acqf_vanilla_bo(
    acq_func: botorch.acquisition.AcquisitionFunction, bounds: torch.Tensor
) -> torch.Tensor:
    """Function to optimize the acquisition function for vanilla Bayesian optimization.

    For instance for expected improvement (botorch.acquisition.analytic.ExpectedImprovement).

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A 2 x D tensor of lower and upper bounds for each column of X.

    Returns:
        A q x D-dim tensor of generated candidates.
    """
    candidates, _ = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,  # Analytic acquisition function.
        num_restarts=5,
        raw_samples=64,  # Used for initialization heuristic.
        options={},
    )
    # Observe new values.
    new_x = candidates.detach()
    return new_x

def initialize_model(train_x, train_obj, label, state_dict=None):
    # define models for objective and constraint
    if label == "quad":
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=train_x.shape[-1],
                batch_shape=None,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=None,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        model_obj = SingleTaskGP(train_x, train_obj, covar_module=covar_module).to(train_x)
    else:
        model_obj = SingleTaskGP(train_x, train_obj).to(train_x)
    #model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    # load state dict if it is passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj

def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    if NORMALIZE:
        bounds=torch.stack(
            [
                torch.zeros(problem.dim, dtype=dtype, device=device),
                torch.ones(problem.dim, dtype=dtype, device=device),
            ])
    else:
        bounds=problem.bounds

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": ACQUISITION_BATCH_OPTIMIZATION, "maxiter": 200},
    )
    # observe new values 
    if NORMALIZE:
        new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    else:
        new_x = candidates
    exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
    new_obj = exact_obj
    return new_x, new_obj

def optimize_acqf_and_get_observation_random(acq_func, dist):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates = dist.sample(torch.tensor([CANDIDATES_VR, BATCH_SIZE])).to(device = problem.device, dtype = problem.dtype)
    res = acq_func(candidates)
    new_x = candidates[torch.argmax(res)]

    exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
    new_obj = exact_obj
    return new_x, new_obj

def update_random_observations():
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = unnormalize(torch.rand(BATCH_SIZE, problem.dim, device=device, dtype=dtype), bounds=problem.bounds)
    rand_y = objective(rand_x).unsqueeze(-1)
    return rand_x, rand_y