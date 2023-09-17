from typing import Dict, Callable, Union, Optional, List, Any
from abc import ABC, abstractmethod
import numpy as np
import torch
import os
from torch.distributions import Normal
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
import botorch
from module.model import DerivativeExactGPSEModel
from module.environment_api import EnvironmentObjective
from module.acquisition_function import GradientInformation, DownhillQuadratic, initialize_acqf_optimizer, piqExpectedImprovement, QuadratureExploration
from module.plot_script import plot_gp_fit, plot_synthesis_quad, plot_distribution_1D
from module.sampler import get_sampler

from scipy.optimize import minimize_scalar
from torch.distributions.multivariate_normal import MultivariateNormal
from .utils import bounded_bivariate_normal_integral, nearestPD, isPD, EI, log_EI, normalize_distribution
from botorch import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize
from .objective import Objective
import geoopt
import matplotlib.pyplot as plt

from botorch.utils.probability.utils import (
    ndtr as Phi
)
from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

LIST_LABEL = ["random", "SNES", "piqEI", "quad", "qEI", "MPD", "BGA"]

def load_optimizer(label, n_init, objective, dict_parameter, plot_path):
    if label == "qEI":
        optimizer = VanillaBayesianOptimization(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["qEI"], plot_path=plot_path)
    elif label == "piqEI":
        optimizer = PiBayesianOptimization(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["piqEI"], plot_path=plot_path)
    elif label == "quad":
        optimizer = ProbES(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["quad"], plot_path=plot_path)
    elif label == "SNES":
        optimizer = CMAES(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["SNES"], plot_path=plot_path)
    elif label == "random":
        optimizer = RandomSearch(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"])
    return optimizer

## Function to generate initial data, either random in bounds or from domain informed distribution
## Make sure seed common for different algorithm (or maybe)

def generate_data(
        label: str,
        objective: Objective,
        n_init: int,
        distribution: torch.distributions.Distribution = None
        ):

    if label in ["random", "qEI"]:
        train_x = unnormalize(torch.rand(n_init, objective.dim, device=objective.device, dtype=objective.dtype), objective.bounds) ### Change initializer normal or discrete
        train_obj = objective(train_x).unsqueeze(-1)  # add output dimension
        best_observed_value = train_obj.max().item()
        
    elif label in ["piqEI", "quad", "SNES"]:
        train_x = distribution.sample((n_init,)).reshape(-1, objective.dim)
        train_obj = objective(train_x).unsqueeze(-1)  # add output dimension
        best_observed_value = train_obj.max().item()
        
    return train_x, train_obj, best_observed_value

## For now, initial data generation depends on optimizer to take domain information into account
class AbstractOptimizer(ABC):
    """Abstract optimizer class.

    Sets a default optimizer interface.

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        param_args_ignore: Which parameters should not be optimized.
        optimizer_config: Configuration file for the optimizer.
    """

    def __init__(
        self,
        n_init: int,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        param_args_ignore: List[int] = None,
        **optimizer_config: Dict,
    ):
        """Inits the abstract optimizer."""
        # Optionally add batchsize to parameters.
        self.n_init = n_init
        self.dim = objective.dim
        self.params = torch.empty((self.dim, 0))
        self.param_args_ignore = param_args_ignore
        self.objective = objective
        self.iteration = 0

    def __call__(self):
        """Call method of optimizers."""
        self.step()

    @abstractmethod
    def step(self) -> None:
        """One parameter update step."""
        pass

    def plot_synthesis(self) -> None:
        """One parameter update step."""
        pass

class RandomSearch(AbstractOptimizer):
    """Implementation of (augmented) random search.

    Method of the nips paper 'Simple random search of static linear policies is
    competitive for reinforcement learning'.

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        step_size: Step-size for parameter update, named alpha in the paper.
        samples_per_iteration: Number of random symmetric samples before
            parameter update, named N in paper.
        exploration_noise: Exploration distance from current parameters, nu in
            paper.
        standard_deviation_scaling: Scaling of the step-size with standard
            deviation of collected rewards, sigma_R in paper.
        num_top_directions: Number of directions that result in the largest
            rewards, b in paper.
        verbose: If True an output is logged.
        param_args_ignore: Which parameters should not be optimized.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        step_size: float,
        samples_per_iteration: int,
        exploration_noise: float,
        standard_deviation_scaling: bool = False,
        num_top_directions: Optional[int] = None,
        verbose: bool = True,
        param_args_ignore: List[int] = None,
    ):
        """Inits random search optimizer."""
        super(RandomSearch, self).__init__(params_init, objective, param_args_ignore)

        self.params_history_list = [self.params.clone()]
        self.step_size = step_size
        self.samples_per_iteration = samples_per_iteration
        self.exploration_noise = exploration_noise
        self._deltas = torch.empty(self.samples_per_iteration, self.params.shape[-1])

        # For augmented random search V1 and V2.
        self.standard_deviation_scaling = standard_deviation_scaling

        # For augmented random search V1-t and V2-t.
        if num_top_directions is None:
            num_top_directions = self.samples_per_iteration
        self.num_top_directions = num_top_directions

        self.verbose = verbose

    def step(self):
        # 1. Sample deltas.
        torch.randn(*self._deltas.shape, out=self._deltas)
        if self.param_args_ignore is not None:
            self._deltas[:, self.param_args_ignore] = 0.0
        # 2. Scale deltas.
        perturbations = self.exploration_noise * self._deltas
        # 3. Compute rewards
        rewards_plus = torch.tensor(
            [
                self.objective(self.params + perturbation)
                for perturbation in perturbations
            ]
        )
        rewards_minus = torch.tensor(
            [
                self.objective(self.params - perturbation)
                for perturbation in perturbations
            ]
        )
        if self.num_top_directions < self.samples_per_iteration:
            # 4. Using top performing directions.
            args_sorted = torch.argsort(
                torch.max(rewards_plus, rewards_minus), descending=True
            )
            args_relevant = args_sorted[: self.num_top_directions]
        else:
            args_relevant = slice(0, self.num_top_directions)
        if self.standard_deviation_scaling is not None:
            # 5. Perform standard deviation scaling.
            std_reward = torch.cat(
                [rewards_plus[args_relevant], rewards_minus[args_relevant]]
            ).std()
        else:
            std_reward = 1.0

        # 6. Update parameters.
        self.params.add_(
            (rewards_plus[args_relevant] - rewards_minus[args_relevant])
            @ self._deltas[args_relevant],
            alpha=self.step_size / (self.num_top_directions * std_reward),
        )

        # 7. Save new parameters.
        if (type(self.objective._func) is EnvironmentObjective) and (
            self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
            # 8. Perform state normalization update.
            self.objective._func._manipulate_state.apply_update()
        else:
            self.params_history_list.append(self.params.clone())

        if self.verbose:
            print(f"Parameter {self.params.numpy()}.")
            print(
                f"Mean of (b) perturbation rewards {torch.mean(torch.cat([rewards_plus[args_relevant], rewards_minus[args_relevant]])) :.2f}."
            )
            if self.standard_deviation_scaling:
                print(f"Std of perturbation rewards {std_reward:.2f}.")

class CMAES(AbstractOptimizer):
    """CMA-ES: Evolutionary Strategy with Covariance Matrix Adaptation for
    nonlinear function optimization.

    Inspired by the matlab code of https://arxiv.org/abs/1604.00772.
    Hence this function does not implement negative weights, that is, w_i = 0 for i > mu.

    Attributes:
        params_init: Objective parameters initial value.
        objective: Objective function.
        sigma: Coordinate wise standard deviation (step-size).
        maximization: True if objective function is maximized, False if minimized.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        n_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        batch_size: int=1,
        optimizer_config: Dict = None,
        plot_path=None
    ):
        """Inits CMA-ES optimizer."""
        super(CMAES, self).__init__(n_init, objective)
        self.batch_size = batch_size
        self.plot_path = plot_path
        self.params_history_list = []
        self.values_history = []

        ## here xmean is initial mu
        self.xmean = torch.zeros(self.dim, dtype=self.objective.dtype)
        self.maximization = True
        self.sigma = np.sqrt(optimizer_config["var_prior"])
        self.sampling_strategy = optimizer_config["sampling_strategy"]
        self.sampler = get_sampler(self.sampling_strategy, batch_size = batch_size, dim = objective.dim, objective=objective)

        # Strategy parameter setting: Selection.
        self.lambda_ = self.batch_size
        self.mu = self.lambda_ // 2  # Number of parents/points for recombination.
        weights = np.log(self.mu + 0.5) - np.log(range(1, self.mu + 1))
        self.weights = torch.tensor(
            weights / sum(weights), dtype=torch.float64
        )  # Normalize recombination weights array.
        self.mueff = sum(self.weights) ** 2 / sum(
            self.weights ** 2
        )  # Variance-effective size of mu.

        # Strategy parameter setting: Adaption.
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )  # Time constant for cumulation for C.
        self.cs = (self.mueff + 2) / (
            self.dim + self.mueff + 5
        )  # Time constant for cumulation for sigma-/step size control.
        self.c1 = 2 / (
            (self.dim + 1.3) ** 2 + self.mueff
        )  # Learning rate for rank-one update of C.
        self.cmu = (
            2
            * (self.mueff - 2 + 1 / self.mueff)
            / ((self.dim + 2) ** 2 + 2 * self.mueff / 2)
        )  # Learning rate for rank-mu update.
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )  # Damping for sigma.

        # Initialize dynamic (internal) strategy parameters and constant.
        self.ps = torch.zeros(self.dim)  # Evolution path for sigma.
        self.pc = torch.zeros(self.dim)  # Evolution path for C.
        self.B = torch.eye(self.dim)
        self.D = torch.eye(
            self.dim
        )  # Eigendecomposition of C (pos. def.): B defines the coordinate system, diagonal matrix D the scaling.
        self.C = self.B @ self.D ** 2 @ self.D.transpose(0, 1)  # Covariance matrix.
        self.eigeneval = 0  # B and D updated at counteval == 0
        self.chiN = self.dim ** 0.5 * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2)
        )  # Expectation of ||N(0,I)|| == norm(randn(N,1))

        # Generation Loop.
        self.arz = torch.empty((self.dim, self.lambda_))
        self.arx = torch.empty((self.dim, self.lambda_))
        self.arfitness = torch.empty((self.lambda_))
        self.counteval = 0
        self.hs = 0
        self.list_mu, self.list_covar = [self.xmean.detach().clone()], [self.C.detach().clone()]
        
        ## Do one vanilla loop

        # 1. Sampling and evaluating.
        for k in range(self.lambda_):
            # Reparameterization trick for samples.
            self.arz[:, k] = torch.randn(
                (self.dim)
            )  # Standard normally distributed vector.
            self.arx[:, k] = (
                self.xmean + self.sigma * self.B @ self.D @ self.arz[:, k]
            )  # Add mutation.
            self.arfitness[k] = self.objective(self.arx[:, k].unsqueeze(0))
            self.counteval += 1
                # 2. Sort solutions.    
        args = torch.argsort(self.arfitness, descending=self.maximization)

        # 3. Update mean.
        self.xold = self.xmean.clone()
        self.xmean = self.arx[:, args[: self.mu]] @ self.weights  # Recombination.

        # 4. Update evolution paths.
        self.ps = (1 - self.cs) * self.ps + (
            np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        ) * self.B * (1/self.D) * self.B.transpose(0,1) * (self.xmean - self.xold) / self.sigma

        if np.linalg.norm(self.ps) / (
            np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_))
        ) < (1.4 + 2 / (self.dim + 1)):
            self.hs = 1

        self.pc = (1 - self.cc) * self.pc + self.hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * (self.xmean - self.xold) / self.sigma

        # 5. Update covariance matrix.
        artmp = (1/self.sigma) * (self.arx[:, args[: self.mu]]-self.xold.reshape(-1, 1).repeat(1,self.mu))
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1
            * (
                self.pc.view(-1, 1) @ self.pc.view(-1, 1).transpose(0, 1)
                + (1 - self.hs) * self.cc * (2 - self.cc) * self.C
            )
            + self.cmu
            * artmp
            @ torch.diag(self.weights)
            @ artmp.transpose(0, 1)
        )

        # 6. Update step-size sigma.
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # 7. Update B and D from C.
        if (
            self.counteval - self.eigeneval
            > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10
        ):
            self.eigeneval = self.counteval
            self.C = torch.triu(self.C) + torch.triu(self.C, diagonal=1).transpose(
                0, 1
            )  # Enforce symmetry.
            D, self.B = torch.linalg.eigh(
                self.C
            )  # Eigendecomposition, B == normalized eigenvectors.
            self.D = torch.diag(
                torch.sqrt(D.clamp_min(1e-20))
            )  # D contains standard deviations now.

        # Escape flat fitness, or better terminate?
        if self.arfitness[0] == self.arfitness[int(np.ceil(0.7 * self.lambda_)) - 1]:
            self.sigma *= np.exp(0.2 + self.cs / self.damps)

        self.params = self.arx[:, args[0]].view(
            1, -1
        )  # Return the best point of the last generation. Notice that xmean is expected to be even better.

        self.params_history_list.append(self.arx.clone().reshape(-1, self.dim))
        self.values_history.append(self.arfitness.clone().reshape(-1, 1))
        self.list_mu.append(self.xmean.detach().clone())
        self.list_covar.append((self.sigma**2) * self.C.detach().clone())

    def step(self):
        
        train_x = torch.vstack(self.params_history_list)
        train_y = torch.vstack(self.values_history)
        
        self.sampler.update_sampler(train_x=train_x, train_y=train_y, xmean = self.xmean, C=self.sigma**2 * self.C)
        self.arx = self.sampler.sample_batch(self.xmean, self.sigma**2 * self.C)
        self.arfitness = self.objective(self.arx.T)
        self.counteval += self.lambda_

        # 2. Sort solutions.    
        args = torch.argsort(self.arfitness, descending=self.maximization)

        # 3. Update mean.
        self.xold = self.xmean.clone()
        self.xmean = self.arx[:, args[: self.mu]] @ self.weights  # Recombination.

        # 4. Update evolution paths.
        self.ps = (1 - self.cs) * self.ps + (
            np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        ) * self.B * (1/self.D) * self.B.transpose(0,1) * (self.xmean - self.xold) / self.sigma

        if np.linalg.norm(self.ps) / (
            np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_))
        ) < (1.4 + 2 / (self.dim + 1)):
            self.hs = 1

        self.pc = (1 - self.cc) * self.pc + self.hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * (self.xmean - self.xold) / self.sigma

        # 5. Update covariance matrix.
        artmp = (1/self.sigma) * (self.arx[:, args[: self.mu]]-self.xold.reshape(-1, 1).repeat(1,self.mu))
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1
            * (
                self.pc.view(-1, 1) @ self.pc.view(-1, 1).transpose(0, 1)
                + (1 - self.hs) * self.cc * (2 - self.cc) * self.C
            )
            + self.cmu
            * artmp
            @ torch.diag(self.weights)
            @ artmp.transpose(0, 1)
        )

        # 6. Update step-size sigma.
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # 7. Update B and D from C.
        if (
            self.counteval - self.eigeneval
            > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10
        ):
            self.eigeneval = self.counteval
            self.C = torch.triu(self.C) + torch.triu(self.C, diagonal=1).transpose(
                0, 1
            )  # Enforce symmetry.
            D, self.B = torch.linalg.eigh(
                self.C
            )  # Eigendecomposition, B == normalized eigenvectors.
            self.D = torch.diag(
                torch.sqrt(D.clamp_min(1e-20))
            )  # D contains standard deviations now.

        # Escape flat fitness, or better terminate?
        if self.arfitness[0] == self.arfitness[int(np.ceil(0.7 * self.lambda_)) - 1]:
            self.sigma *= np.exp(0.2 + self.cs / self.damps)

        self.params = self.arx[:, args[0]].view(
            1, -1
        )  # Return the best point of the last generation. Notice that xmean is expected to be even better.

        self.params_history_list.append(self.arx.clone().reshape(-1, self.dim))
        self.values_history.append(self.arfitness.clone().reshape(-1, 1))
        self.list_mu.append(self.xmean.detach().clone())
        self.list_covar.append((self.sigma**2) * self.C.detach().clone())
        self.iteration += 1

    def plot_synthesis(self) -> None:
        ## Check whether 1D or 2D
        if self.dim == 1:
            fig, ax = plt.subplots()
            bounds = self.objective.bounds
            lb, up = float(bounds[0][0]), float(bounds[1][0])
            ax.set_xlim(lb, up)
            
            #Plot datapoints and objective
            x_history, value_history = torch.vstack(self.params_history_list).cpu().numpy(),torch.vstack(self.values_history).cpu().numpy()
            ax.scatter(x_history, value_history, color='black', label='Training data')
            ax.scatter(x_history[(-self.batch_size):], value_history[(-self.batch_size):], color='red', label='Last selected points')
            test_x = torch.linspace(lb, up, 200, device=self.objective.device, dtype=self.objective.dtype)
            value_ = (self.objective(test_x.unsqueeze(-1))).flatten()
            ax.plot(test_x.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')

            ## Plot distribution
            distribution = MultivariateNormal(loc=self.list_mu[-1], covariance_matrix=self.list_covar[-1])
            plot_distribution_1D(ax, distribution)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Evolutionary search distribution')
            ax.legend()
            fig.savefig(os.path.join(self.plot_path, f"synthesis_{self.iteration}.png"))
        
        elif self.dim == 2:
            raise NotImplementedError
            fig, ax = plt.subplots()
            ax.scatter(params_history_list.cpu().numpy(), targets.cpu().numpy(), color='black', label='Training data')
            ax.scatter(train_X.cpu().numpy()[(-batch):], targets.cpu().numpy()[(-batch):], color='red', label='Last selected points')
            ax.plot(test_x_unormalized.cpu().numpy(), predictions.mean.cpu().numpy()*float(std_Y) + float(mean_Y), color='blue', label='Predictive mean')
            ax.plot(test_x_unormalized.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')
            ax.fill_between(test_x_unormalized.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='lightblue', alpha=0.5, label='Confidence region')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Gaussian Process Regression')
            ax.legend()

class VanillaBayesianOptimization(AbstractOptimizer):
    """Optimizer class for vanilla Bayesian optimization.

    Vanilla stands for the usage of a classic acquisition function like
    expected improvement.

    Atrributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        Model: Gaussian process model.
        model_config: Configuration dictionary for model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        acquisition_function: BoTorch acquisition function.
        acqf_config: Configuration dictionary acquisition function.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        n_init: int = 5,
        objective: Callable[[torch.Tensor], torch.Tensor] = None,
        batch_size: int = 1,
        optimizer_config: Dict=None,
        plot_path=None
    ):
        """Inits the vanilla BO optimizer."""
        super(VanillaBayesianOptimization, self).__init__(n_init, objective)

        self.batch_size = batch_size
        self.plot_path = plot_path
        # Initialization of training data.
        self.unit_cube = torch.tensor([[0.]*self.objective.dim, [1.]*self.objective.dim], dtype=self.objective.dtype, device=self.objective.device)
        self.train_x, self.train_y, _ = generate_data("qEI", objective=objective, n_init=n_init)
        
        self.params_history_list = [self.train_x.clone()]
        self.values_history = [self.train_y.clone()]

        # Acquistion function and its optimization properties.
        self.qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([optimizer_config["mc_samples"]]))
        
        self.optimize_acqf = initialize_acqf_optimizer(
                                                       bounds=self.unit_cube,
                                                       batch_size=self.batch_size,
                                                       num_restarts=optimizer_config["num_restarts"],
                                                       raw_samples=optimizer_config["raw_samples"],
                                                       batch_acq=optimizer_config["batch_acq"],
                                                       maxiter=200)

    def step(self) -> None:
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=self.train_x.shape[-1],
                batch_shape=None,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=None,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        train_y_init_standardized = standardize(self.train_y)
        self.model = SingleTaskGP(normalize(self.train_x, bounds=self.objective.bounds),
                                  train_y_init_standardized,
                                  covar_module=covar_module).to(self.train_x)
        
        self.acquisition_function = qExpectedImprovement(
                    model=self.model, 
                    best_f=train_y_init_standardized.max(),
                    sampler=self.qmc_sampler
                )
        
        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)

        # Optimize acquistion function and get new observation.
        new_x_normalized = self.optimize_acqf(self.acquisition_function)
        new_x = unnormalize(new_x_normalized, bounds=self.objective.bounds)
        new_y = self.objective(new_x).unsqueeze(-1)

        # Update training points.
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        self.params_history_list.append(new_x.clone())
        self.values_history.append(new_y.clone())
        self.iteration += 1

    def plot_synthesis(self):
        if self.objective.dim == 1:
            fig, ax = plt.subplots()
            bounds = self.objective.bounds
            lb, up = float(bounds[0][0]), float(bounds[1][0])
            ax.set_xlim(lb, up)
            plot_gp_fit(ax, self.model, self.train_x, targets=self.train_y, obj=self.objective, batch=self.batch_size, normalize_flag=True)
            fig.savefig(os.path.join(self.plot_path, f"synthesis_{self.iteration}.png"))
        
class PiBayesianOptimization(AbstractOptimizer):
    """Optimizer class for vanilla Bayesian optimization.

    Vanilla stands for the usage of a classic acquisition function like
    expected improvement.

    Atrributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        Model: Gaussian process model.
        model_config: Configuration dictionary for model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        acquisition_function: BoTorch acquisition function.
        acqf_config: Configuration dictionary acquisition function.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        n_init: int = 5,
        objective: Callable[[torch.Tensor], torch.Tensor] = None,
        batch_size: int = 1,
        optimizer_config: Dict = None,
        plot_path=None
    ):
        """Inits the vanilla BO optimizer."""
        super(PiBayesianOptimization, self).__init__(n_init, objective)

        self.batch_size = batch_size
        self.plot_path = plot_path
        # Initialization of training data.
        
        ## Load parameter distribution TODO Transform distribution with respect to bounds
        BETA, VAR_PRIOR = optimizer_config["beta"], optimizer_config["var_prior"]
        self.beta = BETA
        mean, loc = torch.zeros(objective.dim, device=objective.device, dtype=objective.dtype), VAR_PRIOR*torch.eye(objective.dim, device=objective.device, dtype=objective.dtype)
        self.distribution = MultivariateNormal(mean, loc)
        self.distribution_normalized = normalize_distribution(self.distribution, self.objective.bounds)

        # Initialization of training data.
        self.unit_cube = torch.tensor([[0.]*self.objective.dim, [1.]*self.objective.dim], dtype=self.objective.dtype, device=self.objective.device)
        self.train_x, self.train_y, _ = generate_data("piqEI", objective=objective, n_init=n_init, distribution=self.distribution)
        
        self.params_history_list = [self.train_x.clone()]
        self.values_history = [self.train_y.clone()]
        
        # Acquistion function and its optimization properties.
        self.qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([optimizer_config["mc_samples"]]))
        self.optimize_acqf = initialize_acqf_optimizer(
                                                bounds=self.unit_cube,
                                                batch_size=self.batch_size,
                                                num_restarts=optimizer_config["num_restarts"],
                                                raw_samples=optimizer_config["raw_samples"],
                                                batch_acq=optimizer_config["batch_acq"],
                                                maxiter=200)
    def step(self) -> None:
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=self.train_x.shape[-1],
                batch_shape=None,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=None,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        train_y_init_standardized = standardize(self.train_y)
        self.model = SingleTaskGP(normalize(self.train_x, bounds=self.objective.bounds),
                                    train_y_init_standardized,
                                    covar_module=covar_module).to(self.train_x)
        
        self.acquisition_function = piqExpectedImprovement(
                    model=self.model, 
                    best_f=self.model.train_targets.max(),
                    pi_distrib=self.distribution_normalized,
                    n_iter=self.iteration+1, ## here iteration starts at 0
                    beta=self.beta,
                    sampler=self.qmc_sampler
                )
        
        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)

        # Optimize acquistion function and get new observation.
        new_x_normalized = self.optimize_acqf(self.acquisition_function)
        new_x = unnormalize(new_x_normalized, bounds=self.objective.bounds)
        new_y = self.objective(new_x).unsqueeze(-1)

        # Update training points.
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        self.params_history_list.append(new_x.clone())
        self.values_history.append(new_y.clone())
        self.iteration += 1
    
    def plot_synthesis(self):
        if self.objective.dim == 1:
            fig, ax = plt.subplots()
            bounds = self.objective.bounds
            lb, up = float(bounds[0][0]), float(bounds[1][0])
            ax.set_xlim(lb, up)
            plot_gp_fit(ax, self.model, self.train_x, targets=self.train_y, obj=self.objective, batch=self.batch_size, normalize_flag=True)
            fig.savefig(os.path.join(self.plot_path, f"synthesis_{self.iteration}.png"))

class BayesianGradientAscent(AbstractOptimizer):
    """Optimizer for Bayesian gradient ascent.

    Also called gradient informative Bayesian optimization (GIBO).

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        max_samples_per_iteration: Maximum number of samples that are supplied
            by acquisition function before updating the parameters.
        OptimizerTorch: Torch optimizer to update parameters, e.g. SGD or Adam.
        optimizer_torch_config: Configuration dictionary for torch optimizer.
        lr_schedular: Optional learning rate schedular, mapping iterations to
            learning rates.
        Model: Gaussian process model, has to supply Jacobian information.
        model_config: Configuration dictionary for the Gaussian process model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        bounds: Search bounds for optimization of acquisition function.
        delta: Defines search bounds for optimization of acquisition function
            indirectly by defining it within a distance of delta from the
            current parameter constellation.
        epsilon_diff_acq_value: Difference between acquisition values. Sampling
            of new data points with acquisition function stops when threshold of
            this epsilon value is reached.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        normalize_gradient: Algorithmic extension, normalize the gradient
            estimate with its L2 norm and scale the remaining gradient direction
            with the trace of the lengthscale matrix.
        standard_deviation_scaling: Scale gradient with its variance, inspired
            by an augmentation of random search.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        max_samples_per_iteration: int,
        OptimizerTorch: torch.optim.Optimizer,
        optimizer_torch_config: Optional[Dict],
        lr_schedular: Optional[Dict[int, int]],
        Model: DerivativeExactGPSEModel,
        model_config: Optional[
            Dict[
                str,
                Union[int, float, torch.nn.Module, gpytorch.priors.Prior],
            ]
        ],
        hyperparameter_config: Optional[Dict[str, bool]],
        optimize_acqf: Callable[[GradientInformation, torch.Tensor], torch.Tensor],
        optimize_acqf_config: Dict[str, Union[torch.Tensor, int, float]],
        bounds: Optional[torch.Tensor],
        delta: Optional[Union[int, float]],
        epsilon_diff_acq_value: Optional[Union[int, float]],
        generate_initial_data: Optional[
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]
        ],
        normalize_gradient: bool = False,
        standard_deviation_scaling: bool = False,
        verbose: bool = True,
        wandb_run=None,
    ) -> None:
        """Inits optimizer Bayesian gradient ascent."""
        super(BayesianGradientAscent, self).__init__(params_init, objective)

        self.normalize_gradient = normalize_gradient
        self.standard_deviation_scaling = standard_deviation_scaling

        # Parameter initialization.
        self.params_history_list = [self.params.clone()]
        self.params.grad = torch.zeros_like(self.params)
        self.D = self.params.shape[-1]

        # Torch optimizer initialization.
        self.optimizer_torch = OptimizerTorch([self.params], **optimizer_torch_config)
        self.lr_schedular = lr_schedular
        self.iteration = 0

        # Gradient certainty.
        self.epsilon_diff_acq_value = epsilon_diff_acq_value

        # Model initialization and optional hyperparameter settings.
        if (
            hasattr(self.objective._func, "_manipulate_state")
            and self.objective._func._manipulate_state is not None
        ):
            normalize = self.objective._func._manipulate_state.normalize_params
            unnormalize = self.objective._func._manipulate_state.unnormalize_params
        else:
            normalize = unnormalize = None
        self.model = Model(self.D, normalize, unnormalize, **model_config)
        # Initialization of training data.
        if generate_initial_data is not None:
            train_x_init, train_y_init = generate_initial_data(self.objective)
            self.model.append_train_data(train_x_init, train_y_init)

        if hyperparameter_config["hypers"]:
            hypers = dict(
                filter(
                    lambda item: item[1] is not None,
                    hyperparameter_config["hypers"].items(),
                )
            )
            self.model.initialize(**hypers)
        if hyperparameter_config["no_noise_optimization"]:
            # Switch off the optimization of the noise parameter.
            self.model.likelihood.noise_covar.raw_noise.requires_grad = False

        self.optimize_hyperparamters = hyperparameter_config["optimize_hyperparameters"]

        # Acquistion function and its optimization properties.
        self.acquisition_fcn = GradientInformation(self.model)
        self.optimize_acqf = lambda acqf, bounds: optimize_acqf(
            acqf, bounds, **optimize_acqf_config
        )
        self.bounds = bounds
        self.delta = delta
        self.update_bounds = self.bounds is None

        self.max_samples_per_iteration = max_samples_per_iteration
        self.verbose = verbose
        self.wandb_run = wandb_run

        self.old_f_params = 0
        self.f_params = 0
        self.old_f_reward = 0
        self.f_reward = 0
        self.num_successes = 0
        self.num_moves = 0

        self.alpha_star = 0
        self.sampled_alpha_star = 0

    def log_stats(self, log_rewards=True):
        if self.f_reward > self.old_f_reward:
            self.num_successes += 1
        self.num_moves += 1

        log_dict = {}

        log_dict["iter"] = self.objective._calls

        if log_rewards:
            log_dict["f_reward"] = self.f_reward.item()
            log_dict["f_params"] = self.f_params.item()
            log_dict["r"] = self.num_successes / self.num_moves

        log_dict["mean_constant"] = self.model.mean_module.constant.item()
        log_dict["noise_sd"] = self.model.likelihood.noise.detach().sqrt().item()
        log_dict["outputscale"] = self.model.covar_module.outputscale.item()
        lengthscales = (
            self.model.covar_module.base_kernel.lengthscale.detach()
            .numpy()
            .flatten()
            .tolist()
        )
        for l_ind, l in enumerate(lengthscales):
            log_dict[f"lenghtscale{l_ind}"] = l

        with torch.no_grad():
            self.model.train()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )

            log_dict["mll"] = mll(
                self.model(self.model.train_inputs[0]), self.model.train_targets
            ).item()

            self.model.eval()

        log_dict["alpha*"] = self.alpha_star
        log_dict["sampled alpha*"] = self.sampled_alpha_star

        self.wandb_run.log(log_dict)

    def inspect_acq_func(self, num_samples=1_000_000):
        with torch.no_grad():
            sobol_eng = torch.quasirandom.SobolEngine(dimension=self.params.size(-1))
            samples = sobol_eng.draw(num_samples)
            samples = samples.unsqueeze(1)  # batch_shape x q=1 x d
            samples = (
                2 * self.delta * samples - self.delta
            )  # map to the cube around self.params

            acq_values = self.acquisition_fcn(samples)

        return acq_values.max()

    def step(self) -> None:
        # Sample with new params from objective and add this to train data.
        # Optionally forget old points (if N > N_max).
        self.old_f_params = self.f_params
        self.old_f_reward = self.f_reward
        f_params = self.objective(self.params)
        self.f_params = f_params
        try:
            self.f_reward = [*self.objective._func.timesteps_to_reward.values()][-1]
        except:
            self.f_reward = f_params
        if self.verbose:
            print(f"Reward of parameters theta_(t-1): {f_params.item():.2f}.")
        if self.wandb_run is not None:
            self.log_stats()
        self.model.append_train_data(self.params, f_params)

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
            and self.objective._func._manipulate_state.apply_update() is not None
        ):
            self.objective._func._manipulate_state.apply_update()

        self.model.posterior(
            self.params
        )  # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv)

        self.acquisition_fcn.update_theta_i(self.params)
        # Stay local around current parameters.
        if self.update_bounds:
            self.bounds = torch.tensor([[-self.delta], [self.delta]]) + self.params
        # Only optimize model hyperparameters if N >= N_max.
        # if self.optimize_hyperparamters and (
        #     self.model.N >= self.model.N_max
        # ):  # Adjust hyperparameters
        if self.optimize_hyperparamters:
            # self.model.mean_module.constant = torch.nn.Parameter(torch.tensor(0.))
            # self.model.covar_module.base_kernel.lengthscale = 0.7
            # self.model.covar_module.outputscale = 5

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )
            botorch.fit.fit_gpytorch_model(mll)
            self.model.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.

        self.acquisition_fcn.update_theta_i(self.params)

        acq_value_old = None
        for i in range(self.max_samples_per_iteration):
            # Optimize acquistion function and get new observation.
            new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
            new_y = self.objective(new_x)

            self.alpha_star = acq_value.item()
            # self.sampled_alpha_star = self.inspect_acq_func()
            if self.wandb_run is not None:
                self.log_stats(log_rewards=False)

            # Update training points.
            self.model.append_train_data(new_x, new_y)

            if (
                type(self.objective._func) is EnvironmentObjective
                and self.objective._func._manipulate_state is not None
                and self.objective._func._manipulate_state.apply_update() is not None
            ):
                self.objective._func._manipulate_state.apply_update()

            self.model.posterior(self.params)
            self.acquisition_fcn.update_K_xX_dx()

            # Stop sampling if differece of values of acquired points is smaller than a threshold.
            # Equivalent to: variance of gradient did not change larger than a threshold.
            if self.epsilon_diff_acq_value is not None:
                if acq_value_old is not None:
                    diff = acq_value - acq_value_old
                    if diff < self.epsilon_diff_acq_value:
                        if self.verbose:
                            print(
                                f"Stop sampling after {i+1} samples, since gradient certainty is {diff}."
                            )
                        break
                acq_value_old = acq_value

        self.move(method="step")

        self.iteration += 1

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
        else:
            self.params_history_list.append(self.params.clone())

        if self.verbose:
            posterior = self.model.posterior(self.params)
            print(
                f"theta_t: {self.params_history_list[-1].numpy()} predicted mean {posterior.mvn.mean.item(): .2f} and variance {posterior.mvn.variance.item(): .2f} of f(theta_i)."
            )
            print(
                f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}, outputscale: {self.model.covar_module.outputscale.detach().numpy()},  noise {self.model.likelihood.noise.detach().numpy()}"
            )

    def move(self, method):
        if method == "step":
            with torch.no_grad():
                self.optimizer_torch.zero_grad()
                mean_d, variance_d = self.model.posterior_derivative(self.params)
                params_grad = -mean_d.view(1, self.D)
                if self.normalize_gradient:
                    lengthscale = (
                        self.model.covar_module.base_kernel.lengthscale.detach()
                    )
                    params_grad = (
                        torch.nn.functional.normalize(params_grad) * lengthscale
                    )
                if self.standard_deviation_scaling:
                    params_grad = params_grad / torch.diag(
                        variance_d.view(self.D, self.D)
                    )
                if self.lr_schedular:
                    lr = [
                        v for k, v in self.lr_schedular.items() if k <= self.iteration
                    ][-1]
                    self.params.grad[:] = lr * params_grad  # Define as gradient ascent.
                else:
                    self.params.grad[:] = params_grad  # Define as gradient ascent.
                self.optimizer_torch.step()

        elif method == "mu":
            tmp_params, maximized_mean = botorch.optim.optimize_acqf(
                acq_function=botorch.acquisition.analytic.PosteriorMean(
                    model=self.model
                ),
                bounds=torch.tensor([[-10], [10]]) + self.params.detach(),
                q=1,
                num_restarts=1,
                raw_samples=1,
                batch_initial_conditions=self.params.detach(),
            )
            tmp_params = tmp_params.unsqueeze(0)

            self.params.data = tmp_params

        else:
            raise ValueError("invalid move method")

class MPDOptimizer(AbstractOptimizer):
    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        max_samples_per_iteration: int,
        OptimizerTorch: torch.optim.Optimizer,
        optimizer_torch_config: Optional[Dict],
        lr_schedular: Optional[Dict[int, int]],
        Model: DerivativeExactGPSEModel,
        model_config: Optional[
            Dict[
                str,
                Union[int, float, torch.nn.Module, gpytorch.priors.Prior],
            ]
        ],
        hyperparameter_config: Optional[Dict[str, bool]],
        optimize_acqf: Callable[[GradientInformation, torch.Tensor], torch.Tensor],
        optimize_acqf_config: Dict[str, Union[torch.Tensor, int, float]],
        bounds: Optional[torch.Tensor],
        delta: Optional[Union[int, float]],
        epsilon_diff_acq_value: Optional[Union[int, float]],
        generate_initial_data: Optional[
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]
        ],
        normalize_gradient: bool = False,
        standard_deviation_scaling: bool = False,
        verbose: bool = True,
        wandb_run=None,
    ) -> None:
        """Inits optimizer Bayesian gradient ascent."""
        super().__init__(params_init, objective)

        self.normalize_gradient = normalize_gradient
        self.standard_deviation_scaling = standard_deviation_scaling

        # Parameter initialization.
        self.params_history_list = [self.params.clone()]
        self.params.grad = torch.zeros_like(self.params)
        self.D = self.params.shape[-1]

        # Torch optimizer initialization.
        self.optimizer_torch = OptimizerTorch([self.params], **optimizer_torch_config)
        self.lr_schedular = lr_schedular
        self.iteration = 0

        # Gradient certainty.
        self.epsilon_diff_acq_value = epsilon_diff_acq_value

        # Model initialization and optional hyperparameter settings.
        if (
            hasattr(self.objective._func, "_manipulate_state")
            and self.objective._func._manipulate_state is not None
        ):
            normalize = self.objective._func._manipulate_state.normalize_params
            unnormalize = self.objective._func._manipulate_state.unnormalize_params
        else:
            normalize = unnormalize = None
        self.model = Model(self.D, normalize, unnormalize, **model_config)
        # Initialization of training data.
        if generate_initial_data is not None:
            train_x_init, train_y_init = generate_initial_data(self.objective)
            self.model.append_train_data(train_x_init, train_y_init)

        if hyperparameter_config["hypers"]:
            hypers = dict(
                filter(
                    lambda item: item[1] is not None,
                    hyperparameter_config["hypers"].items(),
                )
            )
            self.model.initialize(**hypers)
        if hyperparameter_config["no_noise_optimization"]:
            # Switch off the optimization of the noise parameter.
            self.model.likelihood.noise_covar.raw_noise.requires_grad = False

        self.optimize_hyperparamters = hyperparameter_config["optimize_hyperparameters"]

        # Acquistion function and its optimization properties.
        self.acquisition_fcn = DownhillQuadratic(self.model)
        self.optimize_acqf = lambda acqf, bounds: optimize_acqf(
            acqf, bounds, **optimize_acqf_config
        )
        self.bounds = bounds
        self.delta = delta
        self.update_bounds = self.bounds is None

        self.max_samples_per_iteration = max_samples_per_iteration
        self.verbose = verbose
        self.wandb_run = wandb_run

        self.old_f_params = 0
        self.f_params = 0
        self.old_f_reward = 0
        self.f_reward = 0
        self.num_successes = 0
        self.num_moves = 0

        self.alpha_star = 0
        self.sampled_alpha_star = 0
        self.p_star = 0.5

        self.num_log_calls = 0

    def log_stats(self, log_rewards=True):
        if self.f_reward > self.old_f_reward:
            self.num_successes += 1
        self.num_moves += 1

        log_dict = {}

        log_dict["iter"] = self.objective._calls

        if log_rewards:
            log_dict["f_reward"] = self.f_reward.item()
            log_dict["f_params"] = self.f_params.item()
            log_dict["r"] = self.num_successes / self.num_moves

        log_dict["mean_constant"] = self.model.mean_module.constant.item()
        log_dict["noise_sd"] = self.model.likelihood.noise.detach().sqrt().item()
        log_dict["outputscale"] = self.model.covar_module.outputscale.item()
        lengthscales = (
            self.model.covar_module.base_kernel.lengthscale.detach()
            .numpy()
            .flatten()
            .tolist()
        )
        for l_ind, l in enumerate(lengthscales):
            log_dict[f"lenghtscale{l_ind}"] = l

        with torch.no_grad():
            self.model.train()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )

            log_dict["mll"] = mll(
                self.model(self.model.train_inputs[0]), self.model.train_targets
            ).item()

            self.model.eval()

        log_dict["alpha*"] = self.alpha_star
        log_dict["p*"] = self.p_star

        self.wandb_run.log(log_dict)

    def inspect_acq_func(self, num_samples=1_000_000, bound_multiplier=1):
        # with torch.no_grad():
        #     sobol_eng = torch.quasirandom.SobolEngine(dimension=self.params.size(-1))
        #     samples = sobol_eng.draw(num_samples)
        #     samples = 2 * self.delta * samples - self.delta  # map to the cube around self.params
        #
        #     acq_values = []
        #     for sample in samples:
        #         sample = sample.unsqueeze(0)
        #         acq_values.append(self.acquisition_fcn(sample))
        #
        #     acq_values = torch.cat(acq_values, dim=0)
        #
        # return acq_values.max()

        _, acq_value = botorch.optim.optimize_acqf(
            self.acquisition_fcn,
            bound_multiplier * torch.tensor([[-self.delta], [self.delta]])
            + self.params,
            q=1,
            num_restarts=5,
            raw_samples=num_samples,
            options={"nonnegative": True, "batch_limit": 5},
            return_best_only=True,
            sequential=False,
        )

        return acq_value

    def most_likely_uphill_direction(self, cand_params):
        pred_grad_mean, pred_covar = self.model.posterior_derivative(cand_params)
        pred_grad_L = psd_safe_cholesky(pred_covar).unsqueeze(0)

        best_direction = (
            torch.cholesky_solve(pred_grad_mean.unsqueeze(-1), pred_grad_L)
            .squeeze(-1)
            .squeeze(0)
        )
        best_direction = torch.nn.functional.normalize(best_direction)

        uphill_probability = Normal(0, 1).cdf(
            torch.matmul(best_direction, pred_grad_mean.transpose(-1, -2)).sqrt()
        )

        return best_direction, uphill_probability

    def step(self) -> None:
        # if self.objective._calls == 400:
        #     torch.save(self.params, "params.pt")
        #     torch.save(self.model.train_inputs[0], "train_x.pt")
        #     torch.save(self.model.train_targets, "train_y.pt")
        #     torch.save(self.model.state_dict(), "model_state.pt")
        #     torch.save(self.model.mean_module.constant.detach(), "mean_constant.pt")
        #     torch.save(self.model.covar_module.outputscale.detach(), "outputscale.pt")
        #     torch.save(self.model.covar_module.base_kernel.lengthscale.detach(), "lengthscale.pt")
        #     torch.save(self.model.likelihood.noise.detach(), "noise.pt")
        #
        #     from IPython.core.debugger import set_trace
        #     set_trace()

        # Sample with new params from objective and add this to train data.
        # Optionally forget old points (if N > N_max).
        self.old_f_params = self.f_params
        self.old_f_reward = self.f_reward
        f_params = self.objective(self.params)
        self.f_params = f_params
        try:
            self.f_reward = [*self.objective._func.timesteps_to_reward.values()][-1]
        except:
            self.f_reward = f_params
        if self.verbose:
            print(f"Reward of parameters theta_(t-1): {f_params.item():.2f}.")
        if self.wandb_run is not None:
            self.log_stats()
        self.model.append_train_data(self.params, f_params)

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
            and self.objective._func._manipulate_state.apply_update() is not None
        ):
            self.objective._func._manipulate_state.apply_update()

        self.model.posterior(
            self.params
        )  # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv)

        self.acquisition_fcn.update_theta_i(self.params)
        # Stay local around current parameters.
        if self.update_bounds:
            self.bounds = torch.tensor([[-self.delta], [self.delta]]) + self.params
        # Only optimize model hyperparameters if N >= N_max.
        # if self.optimize_hyperparamters and (
        #     self.model.N >= self.model.N_max
        # ):  # Adjust hyperparameters
        if self.optimize_hyperparamters:
            # self.model.mean_module.constant = torch.nn.Parameter(torch.tensor(0.))
            # self.model.covar_module.base_kernel.lengthscale = 0.7
            # self.model.covar_module.outputscale = 5

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )
            botorch.fit.fit_gpytorch_model(mll)
            # self.model = train_model(self.model)
            with gpytorch.settings.cholesky_jitter(1e-1):
                self.model.posterior(
                    self.params
                )  # Call this to update prediction strategy of GPyTorch.

        self.acquisition_fcn.update_theta_i(self.params)

        for i in range(self.max_samples_per_iteration):
            # Optimize acquistion function and get new observation.
            new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
            new_y = self.objective(new_x)

            self.alpha_star = acq_value.item()

            # Update training points.
            self.model.append_train_data(new_x, new_y)

            with torch.no_grad():
                _, uphill_probability = self.most_likely_uphill_direction(
                    self.params.detach()
                )
                self.p_star = uphill_probability.item()

            if self.wandb_run is not None:
                self.log_stats(log_rewards=False)

            if (
                type(self.objective._func) is EnvironmentObjective
                and self.objective._func._manipulate_state is not None
                and self.objective._func._manipulate_state.apply_update() is not None
            ):
                self.objective._func._manipulate_state.apply_update()

            self.model.posterior(self.params)
            self.acquisition_fcn.update_K_xX_dx()

        self.move(method="iter")

        self.iteration += 1

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
        else:
            self.params_history_list.append(self.params.clone())

        if self.verbose:
            posterior = self.model.posterior(self.params)
            print(
                f"theta_t: {self.params_history_list[-1].numpy()} predicted mean {posterior.mvn.mean.item(): .2f} and variance {posterior.mvn.variance.item(): .2f} of f(theta_i)."
            )
            print(
                f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}, outputscale: {self.model.covar_module.outputscale.detach().numpy()},  noise {self.model.likelihood.noise.detach().numpy()}"
            )

    def step_old(self) -> None:
        # Sample with new params from objective and add this to train data.
        # Optionally forget old points (if N > N_max).
        self.old_f_params = self.f_params
        self.old_f_reward = self.f_reward
        f_params = self.objective(self.params)
        self.f_params = f_params
        try:
            self.f_reward = [*self.objective._func.timesteps_to_reward.values()][-1]
        except:
            self.f_reward = f_params
        if self.verbose:
            print(f"Reward of parameters theta_(t-1): {f_params.item():.2f}.")
        if self.wandb_run is not None:
            self.log_stats()
        self.model.append_train_data(self.params, f_params)

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
            and self.objective._func._manipulate_state.apply_update() is not None
        ):
            self.objective._func._manipulate_state.apply_update()

        self.model.posterior(
            self.params
        )  # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv)

        self.acquisition_fcn.update_theta_i(self.params)
        # Stay local around current parameters.
        if self.update_bounds:
            self.bounds = torch.tensor([[-self.delta], [self.delta]]) + self.params
        # Only optimize model hyperparameters if N >= N_max.
        if self.optimize_hyperparamters and (
            self.model.N >= self.model.N_max
        ):  # Adjust hyperparameters
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )
            botorch.fit.fit_gpytorch_model(mll)
            self.model.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.

        self.acquisition_fcn.update_theta_i(self.params)

        for i in range(self.max_samples_per_iteration):
            # Optimize acquistion function and get new observation.
            new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
            new_y = self.objective(new_x)

            self.alpha_star = acq_value.item()
            log_sampled_alpha = self.objective._calls > self.num_log_calls + 100
            if log_sampled_alpha:
                self.num_log_calls += 100
                self.sampled_alpha_star = self.inspect_acq_func()
                if self.sampled_alpha_star < self.alpha_star:
                    self.sampled_alpha_star = self.alpha_star

            # Update training points.
            self.model.append_train_data(new_x, new_y)

            with torch.no_grad():
                _, uphill_probability = self.most_likely_uphill_direction(
                    self.params.detach()
                )
                self.p_star = uphill_probability.item()

            if log_sampled_alpha and (self.wandb_run is not None):
                log_dict = {}
                log_dict["iter"] = self.objective._calls
                log_dict["64 alpha*"] = self.alpha_star
                log_dict["1mil alpha*"] = self.sampled_alpha_star

                self.wandb_run.log(log_dict)

            if (
                type(self.objective._func) is EnvironmentObjective
                and self.objective._func._manipulate_state is not None
                and self.objective._func._manipulate_state.apply_update() is not None
            ):
                self.objective._func._manipulate_state.apply_update()

            self.model.posterior(self.params)
            self.acquisition_fcn.update_K_xX_dx()

        self.move(method="iter")

        self.iteration += 1

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
        else:
            self.params_history_list.append(self.params.clone())

        if self.verbose:
            posterior = self.model.posterior(self.params)
            print(
                f"theta_t: {self.params_history_list[-1].numpy()} predicted mean {posterior.mvn.mean.item(): .2f} and variance {posterior.mvn.variance.item(): .2f} of f(theta_i)."
            )
            print(
                f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}, outputscale: {self.model.covar_module.outputscale.detach().numpy()},  noise {self.model.likelihood.noise.detach().numpy()}"
            )

    def move(self, method):
        if method == "step":
            with torch.no_grad():
                self.optimizer_torch.zero_grad()
                mean_d, variance_d = self.model.posterior_derivative(self.params)
                params_grad = -mean_d.view(1, self.D)
                if self.normalize_gradient:
                    lengthscale = (
                        self.model.covar_module.base_kernel.lengthscale.detach()
                    )
                    params_grad = (
                        torch.nn.functional.normalize(params_grad) * lengthscale
                    )
                if self.standard_deviation_scaling:
                    params_grad = params_grad / torch.diag(
                        variance_d.view(self.D, self.D)
                    )
                if self.lr_schedular:
                    lr = [
                        v for k, v in self.lr_schedular.items() if k <= self.iteration
                    ][-1]
                    self.params.grad[:] = lr * params_grad  # Define as gradient ascent.
                else:
                    self.params.grad[:] = params_grad  # Define as gradient ascent.
                self.optimizer_torch.step()

        elif method == "mu":
            tmp_params, maximized_mean = botorch.optim.optimize_acqf(
                acq_function=botorch.acquisition.analytic.PosteriorMean(
                    model=self.model
                ),
                bounds=torch.tensor([[-10], [10]]) + self.params.detach(),
                q=1,
                num_restarts=1,
                raw_samples=1,
                batch_initial_conditions=self.params.detach(),
            )
            tmp_params = tmp_params.unsqueeze(0)

            self.params.data = tmp_params

        elif method == "iter":
            with torch.no_grad():
                tmp_params = self.params.detach().clone()

                v_star, p_star = self.most_likely_uphill_direction(tmp_params)
                num_iters = 0
                while p_star >= 0.65 and num_iters <= 10_000:
                    if False:
                        # print("at", tmp_params.detach().numpy().flatten())
                        # print("direction", v_star.detach().numpy().flatten())
                        print("p", p_star.item())
                        print(tmp_params)
                        print()

                    tmp_params += v_star.squeeze(0) * 0.01
                    v_star, p_star = self.most_likely_uphill_direction(tmp_params)
                    num_iters += 1

                self.params.data = tmp_params

        else:
            raise ValueError("invalid move method")

class FiniteDiffGradientAscentOptimizer(AbstractOptimizer):
    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        eta: float = 0.01,
        epsilon: float = 1e-6,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(params_init, objective)

        self.params_history_list = [self.params.clone()]
        self.iteration = 0

        self.eta = eta
        self.epsilon = epsilon
        self.wandb_run = wandb_run

        self.old_f_params = 0
        self.f_params = 0
        self.old_f_reward = 0
        self.f_reward = 0
        self.num_successes = 0
        self.num_moves = 0

    def log_stats(self):
        if self.f_reward > self.old_f_reward:
            self.num_successes += 1
        self.num_moves += 1

        log_dict = {}

        log_dict["iter"] = self.objective._calls

        log_dict["f_reward"] = self.f_reward.item()
        log_dict["f_params"] = self.f_params.item()
        log_dict["r"] = self.num_successes / self.num_moves

        self.wandb_run.log(log_dict)

    def step(self) -> None:
        self.old_f_params = self.f_params
        self.old_f_reward = self.f_reward
        f_params = self.objective(self.params)
        self.f_params = f_params
        try:
            self.f_reward = [*self.objective._func.timesteps_to_reward.values()][-1]
        except:
            self.f_reward = f_params
        if self.wandb_run is not None:
            self.log_stats()

        dim = self.params.size(-1)

        grad = torch.zeros(1, dim)

        for j in range(dim):
            x_left = self.params.detach().clone()
            x_left[0, j] -= self.epsilon

            x_right = self.params.detach().clone()
            x_right[0, j] += self.epsilon

            func_left = self.objective(x_left)
            func_right = self.objective(x_right)
            grad[0, j] = (func_right - func_left) / self.epsilon / 2

        with torch.no_grad():
            self.params += self.eta * grad

        self.iteration += 1

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
        else:
            self.params_history_list.append(self.params.clone())

class FiniteDiffGradientAscentOptimizerv2(FiniteDiffGradientAscentOptimizer):
    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        OptimizerTorch: torch.optim.Optimizer,
        optimizer_torch_config: Optional[Dict],
        lr_schedular: Optional[Dict[int, int]],
        normalize_gradient: bool = False,
        epsilon: float = 1e-6,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(params_init, objective)

        self.normalize_gradient = normalize_gradient

        # Parameter initialization.
        self.params_history_list = [self.params.clone()]
        self.params.grad = torch.zeros_like(self.params)
        self.D = self.params.shape[-1]

        # Torch optimizer initialization.
        self.optimizer_torch = OptimizerTorch([self.params], **optimizer_torch_config)
        self.lr_schedular = lr_schedular
        self.iteration = 0

        self.wandb_run = wandb_run

        self.old_f_params = 0
        self.f_params = 0
        self.old_f_reward = 0
        self.f_reward = 0
        self.num_successes = 0
        self.num_moves = 0

    def step(self) -> None:
        self.old_f_params = self.f_params
        self.old_f_reward = self.f_reward
        f_params = self.objective(self.params)
        self.f_params = f_params
        try:
            self.f_reward = [*self.objective._func.timesteps_to_reward.values()][-1]
        except:
            self.f_rewrad = f_params
        if self.wandb_run is not None:
            self.log_stats()

        dim = self.params.size(-1)

        grad = torch.zeros(1, dim)

        for j in range(dim):
            x_left = self.params.detach().clone()
            x_left[0, j] -= self.epsilon

            x_right = self.params.detach().clone()
            x_right[0, j] += self.epsilon

            func_left = self.objective(x_left)
            func_right = self.objective(x_right)
            grad[0, j] = (func_right - func_left) / self.epsilon / 2

        with torch.no_grad():
            self.optimizer_torch.zero_grad()

            if self.normalize_gradient:
                grad = torch.nn.functional.normalize(grad)
            if self.lr_schedular:
                lr = [v for k, v in self.lr_schedular.items() if k <= self.iteration][
                    -1
                ]
                self.params.grad[:] = lr * -grad  # Define as gradient ascent.
            else:
                self.params.grad[:] = -grad  # Define as gradient ascent.
            self.optimizer_torch.step()

        self.iteration += 1

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
        else:
            self.params_history_list.append(self.params.clone())

## TODO Be careful computational graphs when using .copy() as not deleted and can result in using too much memory
class ProbES(AbstractOptimizer):
    def __init__(
        self,
        n_init: int = 5,
        objective: Callable[[torch.Tensor], torch.Tensor] = None,
        batch_size: int = 1,
        optimizer_config: Dict = None,
        plot_path=None
    ):
        
        super(ProbES, self).__init__(n_init, objective)

        self.batch_size = batch_size
        self.plot_path = plot_path
        if not os.path.exists(os.path.join(self.plot_path, "fitgp")):
            os.makedirs(os.path.join(self.plot_path, "fitgp"))
        ## Load parameter distribution
        mu, var = 0., optimizer_config["var_prior"] ## TODO option for different starting point
        mean, loc = mu*torch.ones(objective.dim, device=objective.device, dtype=objective.dtype), var*torch.eye(objective.dim, device=objective.device, dtype=objective.dtype)
        self.distribution = MultivariateNormal(mean, loc)

        self.train_x, self.train_y, _ = generate_data("quad", objective=objective, n_init = n_init, distribution=self.distribution)
        #self.params = self.train_x.clone()
        self.params_history_list = [self.train_x.clone()]
        self.values_history = [self.train_y.clone()]
        self.d = self.objective.dim
        
        self.lr = optimizer_config["lr"]
        self.policy=optimizer_config["line_search"]
        self.c1=optimizer_config["c1"]
        self.c2=optimizer_config["c2"]
        self.t_max=optimizer_config["t_max"]
        self.budget=optimizer_config["budget"]
        self.manifold=optimizer_config["manifold"]
        self.gradient_direction = optimizer_config["gradient_direction"]
        # Assert wolfe condition of parameters
        assert 0 <= self.c1
        assert self.c1 < self.c2
        assert self.c2 <= 1

        ## Create Manifold
        if self.manifold:
            euclidean = geoopt.manifolds.Euclidean()
            spd = geoopt.manifolds.SymmetricPositiveDefinite()
            self.manifold = geoopt.manifolds.ProductManifold((euclidean, objective.dim), (spd, (objective.dim,objective.dim)))
        else:
            self.manifold = geoopt.manifolds.ProductManifold((euclidean, objective.dim), (euclidean, (objective.dim,objective.dim)))

        ## Define criterion
        if self.policy == "ei": # Change for multivariate gaussian 
            self.optimizer = geoopt.optim.RiemannianAdam((self.param,), lr=1e-2)
        
        elif self.policy in ["wolfe", "armijo"]:
            if self.policy == "wolfe":
                self.criterion = self.wolfe_criterion
            elif self.policy == "armijo":
                self.criterion = self.armijo_criterion
            
        ## TODO Make option to make it an optimization pb instead of samples
        self.optimize_acqf = initialize_acqf_optimizer(type="random", candidate_vr=optimizer_config["candidates_vr"], batch_size=batch_size)

        ## Register mu and covariance history
        self.list_mu, self.list_covar = [self.distribution.loc.detach().clone()], [self.distribution.covariance_matrix.detach().clone()]

        ## Create grads points for autodiff
        mu, covar = self.distribution.loc, self.distribution.covariance_matrix
        self.manifold_point = geoopt.ManifoldTensor(torch.cat((mu, covar.flatten())), manifold=self.manifold)
        self.manifold_point.requires_grad = True
        self.param = geoopt.ManifoldParameter(self.manifold_point)

        ### Make model
        # Model initialization and optional hyperparameter settings.
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=self.train_x.shape[-1],
                batch_shape=None,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=None,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        train_y_init_standardized = standardize(self.train_y)
        self.model = SingleTaskGP(self.train_x, train_y_init_standardized, covar_module=covar_module).to(self.train_x)

        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)

        # Optimize acquistion function and get new observation.
        self.acquisition_function = QuadratureExploration(
            model=self.model,
            distribution=self.distribution)
        
    def wolfe_criterion(self, t):
        target_point = self.manifold.expmap(self.manifold_point, t*self.d_manifold)
        target_point[1] = max(target_point[1], 1e-14)
        mean_joint, covar_joint = self.compute_joint_distribution(self.manifold.take_submanifold_value(target_point, 0), self.manifold.take_submanifold_value(target_point, 1))
        mean_joint = -mean_joint
        return -float(self.compute_wolfe(mean_joint, covar_joint, t))
    
    def armijo_criterion(self, t):
        target_point = self.manifold.expmap(self.manifold_point, t*self.d_manifold)
        mean_joint, covar_joint = self.compute_joint_distribution_first_order(self.manifold.take_submanifold_value(target_point, 0), self.manifold.take_submanifold_value(target_point, 1))
        mean_joint = -mean_joint
        return -float(self.compute_armijo(mean_joint, covar_joint, t).detach().clone().cpu())
    
    def step(self):
        """Make a gradient step toward"""

        # Extract model quantities
        self.train_X = self.model.train_inputs[0]
        self.noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(self.train_X.shape[0], dtype=self.train_X.dtype, device=self.train_X.device)
        self.gp_covariance = ((torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone()))**2).detach().clone()
        self.outputscale = self.model.covar_module.outputscale.detach().clone()
        self.K_X_X = (self.model.covar_module(self.train_X) + self.noise_tensor).evaluate().detach().clone()
        self.mean_constant = self.model.mean_module.constant.detach().clone()
        self.inverse_data_covar_y = torch.linalg.solve(self.K_X_X, self.model.train_targets - self.mean_constant)
        self.inverse_data = torch.linalg.inv(self.K_X_X)

        #### Compute Quadrature
        self.covariance_gp_distr = self.gp_covariance + self.distribution.covariance_matrix
        self.constant = (self.model.covar_module.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()
        self.t_1X = self.constant * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.covariance_gp_distr).log_prob(self.train_X))
        self.R_11 = self.constant * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.covariance_gp_distr + self.distribution.covariance_matrix).log_prob(self.distribution.loc))
        self.inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, self.t_1X) #Independant of t!..
        self.mean_1 = self.mean_constant + (self.t_1X.T @ self.inverse_data_covar_y)
        self.var_1 = self.R_11 - self.t_1X.T @ self.inverse_data_covar_t1X

        ## Compute direction (sampled, expected , mdp ect...).
        if self.gradient_direction == "expected":
            m, _ = self._quadrature(self.manifold.take_submanifold_value(self.manifold_point, 0), self.manifold.take_submanifold_value(self.manifold_point, 1))
            m.backward()
            self.d_manifold = self.manifold_point.grad
            self.d_mu, self.d_epsilon = self.manifold.take_submanifold_value(self.d_manifold, 0), self.manifold.take_submanifold_value(self.d_manifold, 1)

        elif self.gradient_direction == "sampled": ## TODO implement the sampled gradient (require running backprop on v as well)
            raise NotImplementedError
            m, v = self._quadrature(self.manifold.take_submanifold_value(self.manifold_point, 0), self.manifold.take_submanifold_value(self.manifold_point, 1))
            m.backward()
            v.backward()
            s = torch.normal(0, 1, size=(1,), device = self.train_X.device, dtype=self.train_X.dtype)
            self.grad_direction = self.manifold_point.grad + s
            self.d_mu = self.manifold.take_submanifold_value(self.manifold_point.grad, 0) +s*self.manifold.take_submanifold_value(self.manifold_point.grad, 0)

            self.d_epsilon = self.manifold.take_submanifold_value(self.manifold_point.grad, 1) + s*self.manifold.take_submanifold_value(self.manifold_point.grad, 1)
        else:
            raise NotImplementedError
        
        ## Line search mehod;
        if self.policy == "constant":
            self.t_update = self.lr
        elif self.policy in ["wolfe", "armijo"]:
            ## Precompute quantities for method of lines
            Pi_1_1 = self.gp_covariance + 2*self.distribution.covariance_matrix
            self.Pi_inv_1_1 = torch.linalg.inv(Pi_1_1)
            self.fourth_order_Pi1_Pi1 = torch.einsum("ij,kl->ijkl", self.Pi_inv_1_1, self.Pi_inv_1_1)

            # Compute tau matrix
            self.diff_data = (self.train_X - self.distribution.loc) # Nxd
            self.normal_data_unsqueezed = torch.unsqueeze(self.t_1X, -1).repeat(1,self.d)
            self.Tau_mu1 = torch.linalg.solve(self.covariance_gp_distr, self.diff_data * self.normal_data_unsqueezed, left=False)
            
            self.normal_data_unsqueezed = torch.unsqueeze(self.normal_data_unsqueezed, -1).repeat(1,1,self.d)
            self.outer_prod = torch.bmm(torch.unsqueeze(self.diff_data, -1), torch.unsqueeze(self.diff_data, 1))
            self.outer_prod = (self.outer_prod - self.covariance_gp_distr) * self.normal_data_unsqueezed
            self.Tau_epsilon1 = 0.5 * torch.linalg.solve(self.covariance_gp_distr, torch.linalg.solve(self.covariance_gp_distr, self.outer_prod, left=False))
            
            self.R11_prime_epsilon = -0.5*self.R_11*self.Pi_inv_1_1
            self.R1_prime_1_prime_mu_mu = self.Pi_inv_1_1 * self.R_11
            self.R1_prime_1_prime_epsilon_epsilon = (self.fourth_order_Pi1_Pi1 + torch.einsum("ijkl->ikjl" ,self.fourth_order_Pi1_Pi1) + torch.einsum("ijkl->iljk" ,self.fourth_order_Pi1_Pi1)) * self.R_11

            self.mean_prime_1 = (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_y)).sum()
            self.cov_1_1_prime = (self.d_epsilon*self.R11_prime_epsilon).sum() - (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_t1X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_t1X)).sum()
            self.var_prime_1 = self.d_mu @ self.R1_prime_1_prime_mu_mu @ self.d_mu \
                        + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * self.R1_prime_1_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()
            self.t_update = minimize_scalar(self.criterion).x
        
        ## TODO if manifold is euclidean need to project back to semi definite matrices
        self.manifold_point = self.manifold.expmap(self.manifold_point, self.t_update*self.manifold_point.grad)
        mu_target, covar_target = self.manifold.take_submanifold_value(self.manifold_point, 0), self.manifold.take_submanifold_value(self.manifold_point, 1)
        covar_target = torch.max(covar_target, torch.tensor([[1e-14]], device = self.objective.device, dtype = self.objective.dtype))
        self.distribution = MultivariateNormal(mu_target, covar_target)

        """
        Check the grad of mu target
        """

        # Optimize acquistion function and get new observation.
        self.acquisition_function = QuadratureExploration(
                    model=self.model,
                    distribution=self.distribution)
        
        new_x = self.optimize_acqf(acq_func= self.acquisition_function, dist=self.distribution)
        new_y = self.objective(new_x).unsqueeze(-1)

        # Update training points.
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        
        self.list_mu.append(self.distribution.loc.detach().clone())
        self.list_covar.append(self.distribution.covariance_matrix.detach().clone())

        ## Create grads points for autodiff
        mu, covar = self.distribution.loc, self.distribution.covariance_matrix
        self.manifold_point = geoopt.ManifoldTensor(torch.cat((mu, covar.flatten())), manifold=self.manifold)
        self.manifold_point.requires_grad = True
        self.param = geoopt.ManifoldParameter(self.manifold_point)

        ### Make model
        # Model initialization and optional hyperparameter settings.
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=self.train_x.shape[-1],
                batch_shape=None,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=None,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        train_y_init_standardized = standardize(self.train_y)
        self.model = SingleTaskGP(self.train_x, train_y_init_standardized, covar_module=covar_module).to(self.train_x) ## Can input subset of the dataset

        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)
        self.params_history_list.append(new_x.clone())
        self.values_history.append(new_y.clone())
        self.iteration += 1

    # The rbf kernel is not a Gaussian kernel (a multiplicative constant is missing)
    def _quadrature(self, mean, covariance):
        ### quadrature function for optimization and gradient differentiation
        covariance_gp_distr = self.gp_covariance + covariance
        constant = (self.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()
        
        t_1X = constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = covariance_gp_distr).log_prob(self.train_X))
        R_11 = constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = covariance_gp_distr + covariance).log_prob(mean))

        inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, t_1X) #Independant of t!..
        mean_1 = self.mean_constant + (t_1X.T @ self.inverse_data_covar_y)
        var_1 = R_11 - t_1X.T @ inverse_data_covar_t1X
        
        return mean_1, var_1

    # The rbf kernel is not a Gaussian kernel (a multiplicative constant is missing)
    def loss_ei(self):
        mu, Epsilon = self.manifold.take_submanifold_value(self.param, 0), self.manifold.take_submanifold_value(self.param, 1)
        mean, covar = self.compute_target_distribution_zero_order(mu, Epsilon)
        return -log_EI(mean, torch.sqrt(covar), self.best_f)

    def gradient_direction(self, sample = False):
        ## Check output scale
        self.gradient_m_mu = self.Tau_mu1.T @ self.inverse_data_covar_y
        self.gradient_m_epsilon = self.Tau_epsilon1.T @ self.inverse_data_covar_y

        ### Sample a gradient or take the expected gradient direction
        if sample:
            print("Sampling gradient not yet implemented properly")
            self.inverse_data_covar_t = self.inverse_data_covar_t1X
            self.R_prime = -0.5 * self.theta * torch.linalg.inv(self.covariance_gp_distr + self.distribution.covariance_matrix) * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.covariance_gp_distr).log_prob(self.distribution.loc))
            self.gradient_v_mu = self.Tau_mu1.T @ self.inverse_data_covar_t
            self.gradient_v_epsilon = self.R_prime - self.Tau_epsilon.T @ self.inverse_data_covar_t
            s = torch.normal(0, 1, size=(1,), device = self.train_X.device, dtype=self.train_X.dtype)
            self.d_mu = self.theta*(self.gradient_m_mu + s*self.gradient_v_mu)
            self.d_epsilon = self.theta*(self.gradient_m_epsilon + s*self.gradient_v_epsilon)
        else:
            self.d_mu = self.gradient_m_mu
            self.d_epsilon = self.gradient_m_epsilon
    
        ## Precompute quantities for method of lines
        self.R11_prime_epsilon = -0.5*self.R_11*self.Pi_inv_1_1
        self.R1_prime_1_prime_mu_mu = self.Pi_inv_1_1 * self.R_11
        self.R1_prime_1_prime_epsilon_epsilon = (self.fourth_order_Pi1_Pi1 + torch.einsum("ijkl->ikjl" ,self.fourth_order_Pi1_Pi1) + torch.einsum("ijkl->iljk" ,self.fourth_order_Pi1_Pi1)) * self.R_11

        self.mean_prime_1 = (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_y)).sum()
        self.cov_1_1_prime = (self.d_epsilon*self.R11_prime_epsilon).sum() - (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_t1X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_t1X)).sum()
        self.var_prime_1 = self.d_mu @ self.R1_prime_1_prime_mu_mu @ self.d_mu \
                    + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * self.R1_prime_1_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()

    def compute_armijo(self, mean_joint, covar_joint, t):
        A_transform = torch.tensor([[1, self.c1*t, -1]], dtype=self.train_X.dtype, device=self.train_X.device)
        if not isPD(covar_joint):
            return torch.tensor(0.)
        
        mean_armijo = (A_transform @ mean_joint).squeeze(0)
        sigma_armijo = torch.sqrt(A_transform @ covar_joint @ A_transform.T).squeeze(0).squeeze(0)
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        
        return Phi(mean_armijo/sigma_armijo)

    def compute_wolfe(self, mean_joint, covar_joint, t):
        A_transform = torch.tensor([[1, self.c1*t, -1, 0], [0, -self.c2, 0, 1]], dtype=self.train_X.dtype, device=self.train_X.device)
        if not isPD(covar_joint):
            return torch.tensor(0.)
        
        mean_wolfe = A_transform @ mean_joint
        covar_wolfe = A_transform @ covar_joint @ A_transform.T
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        rho = (covar_wolfe[0,1]/torch.sqrt(covar_wolfe[0,0]*covar_wolfe[1,1])).detach().cpu()
        al = -(mean_wolfe[0]/torch.sqrt(covar_wolfe[0,0])).detach().cpu()
        bl = -(mean_wolfe[1]/torch.sqrt(covar_wolfe[1,1])).detach().cpu()
        
        return bounded_bivariate_normal_integral(rho, al, torch.inf, bl, torch.inf)
    
    def compute_p_wolfe(self, t):
        result = self.compute_joint_distribution(t)
        if not result:
            return 0
        mean_joint, covar_joint = result
        mean_joint = -mean_joint ## wolfe is for minimization, we maximize by default
        return self.compute_wolfe(mean_joint, covar_joint, t)
        
    def compute_p_armijo(self, t):
        result = self.compute_joint_distribution_first_order(t)
        if not result:
            return 0
        mean_joint, covar_joint = result
        mean_joint = -mean_joint ## wolfe is for minimization, we maximize by default
        return self.compute_armijo(mean_joint, covar_joint, t)
        
    def compute_joint_distribution_zero_order(self, mu2, Epsilon2):
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # For now assume mu2 and epsilon2 are of shape (b1xb2x....xbk)xd and (b1xb2x....xbk)xdxd (assume no batching)
        # Compute mu and PI

        mu1 = self.distribution.loc
        Epsilon1 = self.distribution.covariance_matrix
        
        if not isPD(Epsilon2):
            return None
        
        Pi_1_2 = self.gp_covariance + Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2


        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))

        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + t2X @ self.inverse_data_covar_y

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - self.t_1X @ inverse_data_covar_t2X
        mean_joint = torch.cat((self.mean_1.unsqueeze(0), mean_2.unsqueeze(0)))
        covar_joint = torch.cat((self.var_1.unsqueeze(0), cov_1_2.unsqueeze(0), cov_1_2.unsqueeze(0), var_2.unsqueeze(0))).reshape(2,2)
        return mean_joint, covar_joint
    
    def compute_target_distribution_zero_order(self, mu, Epsilon):
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # For now assume mu2 and epsilon2 are of shape (b1xb2x....xbk)xd and (b1xb2x....xbk)xdxd (assume no batching)
        # Compute mu and PI
        if not isPD(Epsilon):
            return None
        
        Pi_2_2 = self.gp_covariance + 2*Epsilon


        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu, covariance_matrix = Pi_2_2).log_prob(mu))
        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu, covariance_matrix=Epsilon + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon - self.gp_covariance) * normal_data_unsqueezed2
        
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.mean_constant + t2X @ self.inverse_data_covar_y

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        return mean_2, var_2
    
    def compute_joint_distribution_first_order(self, mu2, Epsilon2):
        """Computes joint distribution (f(0), f'(0), f(t))"""
        # Already changed dCov and Covd here
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        mu1, Epsilon1 = self.distribution.loc, self.distribution.covariance_matrix

        if not isPD(Epsilon2):
            Epsilon2 = nearestPD(Epsilon2)
            return None
        
        Pi_1_2 = self.gp_covariance + Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2
        Pi_inv_1_2 = torch.linalg.inv(Pi_1_2)
        nu = Pi_inv_1_2 @ (mu1 - mu2)
        
        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))
        R12_prime_mu = nu*R12
        R12_prime_epsilon = -0.5*(torch.linalg.inv(Pi_1_2) - nu @ nu.T)*R12
        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + t2X @ self.inverse_data_covar_y

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - self.t_1X @ inverse_data_covar_t2X
        
        product_R12_prime_mu, product_R12_prime_epsilon = (self.d_mu*R12_prime_mu).sum(), (self.d_epsilon*R12_prime_epsilon).sum()
        #R21_prime_mu = - R12_prime_mu and R21_prime_epsilon = R12_prime_epsilon
        cov_2_1_prime = -product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (self.Tau_mu1.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ inverse_data_covar_t2X)).sum()
        
        
        mean_joint = torch.tensor([self.mean_1, self.mean_prime_1, mean_2], dtype=self.train_X.dtype, device=self.train_X.device)
        covar_joint = torch.tensor([[self.var_1, self.cov_1_1_prime, cov_1_2],
                                    [self.cov_1_1_prime, self.var_prime_1, cov_2_1_prime],
                                    [cov_1_2, cov_2_1_prime, var_2]], dtype=self.train_X.dtype, device=self.train_X.device)
        return mean_joint.detach(), covar_joint.detach()

    def compute_joint_distribution(self, mu2, Epsilon2):
        # Already changed dCov and Covd here
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # Compute mu and PI
        mu1, Epsilon1 = self.distribution.loc, self.distribution.covariance_matrix
        
        if not isPD(Epsilon2):
            Epsilon2 = nearestPD(Epsilon2)
            return None
        
        Pi_1_2 = self.gp_covariance + Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2
        Pi_inv_1_2 = torch.linalg.inv(Pi_1_2)
        Pi_inv_2_2 = torch.linalg.inv(Pi_2_2)
        
        nu = Pi_inv_1_2 @ (mu1 - mu2)
        
        third_order_Pi_nu = torch.einsum("i,jk->ijk", nu, Pi_inv_1_2)
        third_order_nu_nu_nu = torch.einsum("i,j,k->ijk", nu, nu, nu)

        fourth_order_Pi_nu_nu = torch.einsum("ij,k,l->ijkl", Pi_inv_1_2, nu, nu)
        fourth_order_nu_nu_nu_nu = torch.einsum("i,j,k,l->ijkl", nu, nu, nu, nu)
        fourth_order_Pi_Pi = torch.einsum("ij,kl->ijkl", Pi_inv_1_2, Pi_inv_1_2)
        fourth_order_Pi2_Pi2 = torch.einsum("ij,kl->ijkl", Pi_inv_2_2, Pi_inv_2_2)
        
        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))
        R12_prime_mu = nu*R12
        R12_prime_epsilon = -0.5*(torch.linalg.inv(Pi_1_2) - nu @ nu.T)*R12
        R1_prime_2_prime_mu_mu = -2*R12_prime_epsilon
        R1_prime_2_prime_mu_epsilon = 0.5*R12*(third_order_Pi_nu + torch.einsum("ijk->jki" ,third_order_Pi_nu) + torch.einsum("ijk->kij" ,third_order_Pi_nu) - third_order_nu_nu_nu)
        R1_prime_2_prime_epsilon_epsilon = 0.25*R12*(fourth_order_nu_nu_nu_nu
                                                    - fourth_order_Pi_nu_nu - torch.einsum("ijkl->ikjl" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->iljk" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jkil" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jlik" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->klij" ,fourth_order_Pi_nu_nu)
                                                    + fourth_order_Pi_Pi + torch.einsum("ijkl->ikjl" ,fourth_order_Pi_Pi) + torch.einsum("ijkl->iljk" ,fourth_order_Pi_Pi))

        R22_prime_epsilon = -0.5*R22*Pi_inv_2_2
        R2_prime_2_prime_mu_mu = Pi_inv_2_2 * R22
        R2_prime_2_prime_epsilon_epsilon = (fourth_order_Pi2_Pi2 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi2_Pi2) + torch.einsum("ijkl->iljk" ,fourth_order_Pi2_Pi2)) * R22

        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)
        Tau2_mu = torch.linalg.solve(Epsilon2 + self.gp_covariance, diff_data2 * normal_data_unsqueezed2, left=False)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        Tau_epsilon2 = 0.5*torch.linalg.solve(Epsilon2 + self.gp_covariance, torch.linalg.solve(Epsilon2 + self.gp_covariance, outer_prod2, left=False))
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + t2X @ self.inverse_data_covar_y
        mean_prime_2 = (self.d_mu * (Tau2_mu.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (Tau_epsilon2.T @ self.inverse_data_covar_y)).sum()

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - self.t_1X @ inverse_data_covar_t2X
        
        # Compute Var 
        cov_2_2_prime = (self.d_epsilon*R22_prime_epsilon).sum() - (self.d_mu * (Tau2_mu.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_t2X)).sum()

        product_R12_prime_mu, product_R12_prime_epsilon = (self.d_mu*R12_prime_mu).sum(), (self.d_epsilon*R12_prime_epsilon).sum()
        cov_1_2_prime = product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (Tau2_mu.T @ self.inverse_data_covar_t1X)).sum() - (self.d_epsilon * (Tau_epsilon2.T @ self.inverse_data_covar_t1X)).sum()
        #R21_prime_mu = - R12_prime_mu and R21_prime_epsilon = R12_prime_epsilon
        cov_2_1_prime = -product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (self.Tau_mu1.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ inverse_data_covar_t2X)).sum()
        
        ## Compute fourth term
        cov_1_prime_2_prime = self.d_mu @ R1_prime_2_prime_mu_mu @ self.d_mu \
                            + 2*((self.d_mu @ R1_prime_2_prime_mu_epsilon) * self.d_epsilon).sum() \
                            + ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R1_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum() \
                            - self.d_mu @ self.Tau_mu1.T @ self.inverse_data @ Tau2_mu @ self.d_mu - (self.d_epsilon*(self.Tau_epsilon1.T @ self.inverse_data @ Tau2_mu @ self.d_mu)).sum() - (self.d_epsilon*(Tau_epsilon2.T @ self.inverse_data @ self.Tau_mu1 @ self.d_mu)).sum() \
                            - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", self.Tau_epsilon1.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()
        
        var_prime_2 = self.d_mu @ R2_prime_2_prime_mu_mu @ self.d_mu \
                    + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R2_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()

        mean_joint = torch.tensor([self.mean_1, self.mean_prime_1, mean_2,  mean_prime_2], dtype=self.train_X.dtype, device=self.train_X.device)
        covar_joint = torch.tensor([[self.var_1, self.cov_1_1_prime, cov_1_2, cov_1_2_prime],
                                    [self.cov_1_1_prime, self.var_prime_1, cov_2_1_prime, cov_1_prime_2_prime],
                                    [cov_1_2, cov_2_1_prime, var_2, cov_2_2_prime],
                                    [cov_1_2_prime, cov_1_prime_2_prime, cov_2_2_prime, var_prime_2]], dtype=self.train_X.dtype, device=self.train_X.device)
        return mean_joint.detach(), covar_joint.detach()

    def plot_synthesis(self, save_path=".", iteration=0):
        plot_synthesis_quad(self, iteration=iteration, save_path=self.plot_path, standardize=True)
    
#### Add scipy zero order optimiser and multi restart
class multistart_scipy(AbstractOptimizer):
    def __init__(
        self,
        n_init: int = 5,
        objective: Callable[[torch.Tensor], torch.Tensor] = None,
        batch_size: int = 1,
        optimizer_config: Dict = None,
        plot_path=None
    ):
        
        super(ProbES, self).__init__(n_init, objective)
        self.batch_size=batch_size
        self.plot_path=plot_path


    def step(self):

        return super().step()

