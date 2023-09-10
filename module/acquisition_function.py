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
        pi_distrib: torch.distributions.Distribution,
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
        
class QuadratureExploration(botorch.acquisition.AnalyticAcquisitionFunction):
    r"""Single-outcome Quadrature bRT.
    Quadrature variance minimization acquisitiion function
    """
    def __init__(
        self,
        model: Model,
        distribution: torch.distributions.MultivariateNormal,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ):
        r"""Single-outcome Quadrature bRT.
        Args:
            model: A fitted single-outcome model.
            distribution: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.distribution=distribution
        self.train_X = self.model.train_inputs[0]
        self.gp_covariance = (torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone()))**2
        self.constant = self.model.covar_module.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Exploration Quadrature value
        """
        batch_size = X.shape[0]
        train_X_batch = torch.unsqueeze(self.train_X, 0).repeat(batch_size, 1, 1)
        X_full = torch.cat((train_X_batch, X), dim= 1)
        noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(X[0].shape[0] + self.train_X.shape[0], dtype=X.dtype, device=X.device)
        t_X = self.constant * torch.exp(torch.distributions.MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.distribution.covariance_matrix + self.gp_covariance).log_prob(X_full))
        v = torch.linalg.solve((self.model.covar_module(X_full) + noise_tensor).evaluate(), t_X)
        return (v * t_X).sum(dim=-1)

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

def initialize_acqf_optimizer(random = False, **kwargs):
    """Acquisition function initializer"""

    def optimize_acqfunction(acq_func):
        """Optimizes the acquisition function, and returns a new candidate."""
        # optimize

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": ACQUISITION_BATCH_OPTIMIZATION, "maxiter": maxiter},
        )
        return candidates
    
    def optimize_acqfunction_random(acq_func, dist):
        """Optimizes the acquisition function via random sampling, and returns a new candidate."""
        # optimize
        candidates = dist.sample(torch.tensor([CANDIDATES_VR, BATCH_SIZE])).to(device = dist.loc.device, dtype = dist.loc.dtype)
        res = acq_func(candidates)
        new_x = candidates[torch.argmax(res)]
        return new_x
    
    if random:
        CANDIDATES_VR = kwargs.get("candidates_vr", 5000)
        BATCH_SIZE = kwargs.get("batch_size", 1)
        return optimize_acqfunction_random
    else:
        bounds = kwargs.get("bounds", None)
        BATCH_SIZE = kwargs.get("batch_size", 1)
        NUM_RESTARTS = kwargs.get("num_restarts", 10)
        RAW_SAMPLES = kwargs.get("raw_samples", 512)
        ACQUISITION_BATCH_OPTIMIZATION = kwargs.get("batch_acq", 5)
        maxiter = kwargs.get("maxiter", 200)
        return optimize_acqfunction