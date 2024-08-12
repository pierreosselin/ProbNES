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
from module.plot_script import plot_synthesis_quad, plot_synthesis_ls
from module.quadrature import Quadrature
from module.linesearch import load_linesearch

from torch.distributions.multivariate_normal import MultivariateNormal
from .utils import bounded_bivariate_normal_integral, nearestPD, isPD, EI, log_EI, _is_in_ellipse
from botorch import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize
from .objective import Objective
import geoopt
import matplotlib.pyplot as plt
import scipy
import wandb


from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from evotorch import Problem
import evotorch.algorithms as evoalgo
import torch
from PIL import Image
from io import BytesIO

LIST_LABEL = ["random", "ES", "piqEI", "probES", "qEI", "MPD", "BGA"]

def load_optimizer(label, n_init, objective, dict_parameter, plot_path):
    if label == "qEI":
        optimizer = VanillaBayesianOptimization(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["qEI"], plot_path=plot_path)
    elif label == "piqEI":
        optimizer = PiBayesianOptimization(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["piqEI"], plot_path=plot_path)
    elif label == "probES":
        optimizer = ProbES(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["probES"], plot_path=plot_path)
    elif label == "ES":
        optimizer = ES(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["ES"], plot_path=plot_path)
    elif label == "random":
        optimizer = Random(n_init=n_init, objective=objective, batch_size=dict_parameter["batch_size"], optimizer_config=dict_parameter["random"], plot_path=plot_path)
    return optimizer

## Function to generate initial data, either random in bounds or from domain informed distribution
## Make sure seed common for different algorithm (or maybe)

def generate_data(
        label: str,
        objective: Objective,
        n_init: int,
        distribution: torch.distributions.Distribution = None
        ):

    if label in ["qEI"]:
        train_x = unnormalize(torch.rand(n_init, objective.dim, device=objective.device, dtype=objective.dtype), objective.bounds) ### Change initializer normal or discrete
        train_obj = objective(train_x).unsqueeze(-1)  # add output dimension
        best_observed_value = train_obj.max().item()
        
    elif label in ["random", "piqEI", "quad", "SNES"]:
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

class Random(AbstractOptimizer):
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
        super(Random, self).__init__(n_init, objective)

        self.batch_size = batch_size
        self.plot_path = plot_path
        # Initialization of training data.
        
        ## Load parameter distribution TODO Transform distribution with respect to bounds
        MEAN, VAR_PRIOR = optimizer_config["mean_prior"], optimizer_config["std_prior"]**2
        mean, covar = MEAN*torch.ones(objective.dim, device=objective.device, dtype=objective.dtype), VAR_PRIOR*torch.eye(objective.dim, device=objective.device, dtype=objective.dtype)
        self.distribution = MultivariateNormal(mean, covar)

        # Initialization of training data.
        # self.unit_cube = torch.tensor([[0.]*self.objective.dim, [1.]*self.objective.dim], dtype=self.objective.dtype, device=self.objective.device)
        self.train_x, self.train_y, _ = generate_data("random", objective=objective, n_init=n_init, distribution=self.distribution)
        
        self.params_history_list = [self.train_x.clone()]
        self.values_history = [self.train_y.clone()]

    def step(self) -> None:

        # Optimize acquistion function and get new observation.
        new_x = self.distribution.rsample((self.batch_size,))
        # new_x_normalized = self.optimize_acqf(self.acquisition_function)
        # new_x = unnormalize(new_x_normalized, bounds=self.objective.bounds)
        new_y = self.objective(new_x).unsqueeze(-1)

        # Update training points.
        self.params_history_list.append(new_x.clone())
        self.values_history.append(new_y.clone())
        self.iteration += 1
    
    def plot_synthesis(self):
        return {"objective": np.max(torch.vstack(self.values_history).cpu().numpy())}

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

class ES(AbstractOptimizer):
    def __init__(
        self,
        n_init: int = 5,
        objective: Callable[[torch.Tensor], torch.Tensor] = None,
        batch_size: int = 1,
        optimizer_config: Dict = None,
        plot_path=None
    ):
        """Inits the vanilla BO optimizer."""
        super(ES, self).__init__(n_init, objective)

        self.batch_size = batch_size
        self.plot_path = plot_path
        self.type = optimizer_config["type"]
        # Create problem
        problem = Problem("max", objective, solution_length=objective.dim, device = objective.device,initial_bounds=objective.bounds, dtype=objective.dtype)

        ## Load parameter distribution TODO Transform distribution with respect to bounds
        self.mean, self.std = optimizer_config["mean_prior"]*torch.ones(objective.dim, device=objective.device, dtype=objective.dtype), torch.tensor([optimizer_config["std_prior"]]*objective.dim, device=objective.device, dtype=objective.dtype)
        if self.type == "SNES":
            self.searcher = evoalgo.SNES(problem, popsize=self.batch_size, center_init = self.mean, stdev_init=self.std)
        elif self.type == "XNES":
            self.searcher = evoalgo.XNES(problem, popsize=self.batch_size, center_init = self.mean, stdev_init=self.std)
        elif self.type == "CMAES":
            self.searcher = evoalgo.CMAES(problem, popsize=self.batch_size, center_init = self.mean, stdev_init=optimizer_config["std_prior"])
        else:
            raise NotImplementedError
        
        self.params_history_list = []
        self.values_history = []
        self.list_mu = []
        self.list_covar = []


        # Initialization of training data.
        if self.type == "CMAES":
            self.values_history.append(self.objective(self.searcher._get_center()).repeat(self.batch_size))
            # self.values_history.append(train_obj.clone())
            self.list_mu.append(self.searcher._get_center())
            self.list_covar.append((self.searcher._get_sigma()**2)*self.searcher.C)
        else:
            self.values_history.append(self.objective(self.searcher._get_mu()).repeat(self.batch_size))
            # self.values_history.append(train_obj.clone())
            self.list_mu.append(self.searcher._get_mu())
            self.list_covar.append(self.searcher._get_sigma()**2)
        self.searcher.run(1)
        train_x, train_obj = self.searcher.population.values, self.searcher.population.evals
        self.params_history_list.append(train_x.clone())
        # self.values_history.append(train_obj.clone())
        

        # np.max(optimizer.objective(torch.vstack(optimizer.list_mu)).cpu().numpy()

    def step(self) -> None:
        if self.type == "CMAES":
            self.values_history.append(self.objective(self.searcher._get_center()).repeat(self.batch_size))
            # self.values_history.append(train_obj.clone())
            self.list_mu.append(self.searcher._get_center())
            self.list_covar.append((self.searcher._get_sigma()**2)*self.searcher.C)
        else:
            self.values_history.append(self.objective(self.searcher._get_mu()).repeat(self.batch_size))
            # self.values_history.append(train_obj.clone())
            self.list_mu.append(self.searcher._get_mu())
            self.list_covar.append(self.searcher._get_sigma()**2)
        self.searcher.run(1)
        train_x, train_obj = self.searcher.population.values, self.searcher.population.evals
        self.params_history_list.append(train_x.clone())
        # self.values_history.append(train_obj.clone())
        
    def plot_synthesis(self):
        """Return the synthesis as a dictionary that can be fed to wandb"""
        batch_samples = self.params_history_list[-1]
        matrix_distances = torch.cdist(batch_samples, batch_samples)
        avg_distance = matrix_distances[torch.triu(torch.ones(matrix_distances.shape), diagonal=1) == 1].mean()
        avg_from_center = torch.linalg.norm((self.list_mu[-1] - self.params_history_list[-1]), dim=1).mean()
        if self.type == "CMAES":
            covariance_matrix = (self.searcher._get_sigma()**2)*self.searcher.C
        else:
            covariance_matrix = self.searcher._distribution.cov
        covar_volume = torch.linalg.det(covariance_matrix)**(1/self.dim)

        if self.objective.dim == 1:
            fig, ax = plt.subplots()
            bounds = self.objective.bounds
            lb, up = float(bounds[0][0]), float(bounds[1][0])
            ax.set_xlim(lb, up)

            # Plot objective
            X_test = torch.linspace(lb, up, 1000).reshape(-1, 1)
            y_test = self.objective(X_test)
            ax.plot(X_test, y_test, color='green', label='True Function')

            # Plot training data
            X = torch.vstack(self.params_history_list).cpu().numpy()
            y = torch.vstack(self.values_history).cpu().numpy()
            ax.scatter(X, y)

            # Plot distribution 
            if self.type == "CMAES":
                mean, std = self.searcher._get_mu().cpu().numpy(), self.searcher._get_sigma().cpu().numpy()
            else:
                mean, std = self.searcher._get_mu().cpu().numpy(), self.searcher._get_sigma().cpu().numpy()
            ax.vlines(x = mean, ymin = min(y_test), ymax = max(y_test), colors = 'red', label = 'Mean distribution')
            ax.vlines(x = mean - 2*std, ymin = min(y_test), ymax = max(y_test), colors = 'red', linestyle='dashed')
            ax.vlines(x = mean + 2*std, ymin = min(y_test), ymax = max(y_test), colors = 'red', linestyle='dashed')
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            # Open image with Pillow
            image = Image.open(buf)
            image = wandb.Image(image)
            return {"Image synthesis info": image,
                    "mean": self.list_mu[-1],
                    "var": self.list_covar[-1],
                    "objective": np.max(torch.vstack(self.values_history).cpu().numpy()),
                    "objective_mean": np.max(self.objective(torch.vstack(self.list_mu)).cpu().numpy()),
                    "current objective_mean": self.objective(self.list_mu[-1]).cpu().numpy(),
                    "Avg distance samples": avg_distance,
                    "Avg distance samples from center": avg_from_center,
                    "Ellipse volume": covar_volume}
    
        else:
            if len(self.list_mu) > 1:
                return {"mean": self.list_mu[-1],
                        "difference mean": torch.linalg.norm(self.list_mu[-1] - self.list_mu[-2]),
                        "difference covar": torch.linalg.norm(self.list_covar[-1] - self.list_covar[-2]),
                        "objective": np.max(torch.vstack(self.values_history).cpu().numpy()),
                        "objective_mean": np.max(self.objective(torch.vstack(self.list_mu)).cpu().numpy()),
                        "current objective_mean": self.objective(self.list_mu[-1]).cpu().numpy(),
                        "Avg distance samples": avg_distance,
                        "Avg distance samples from center": avg_from_center,
                        "Ellipse volume": covar_volume}
            else:
                return {"mean": self.list_mu[-1],
                        "difference mean": 0.,
                        "difference covar": 0.,
                        "objective": np.max(torch.vstack(self.values_history).cpu().numpy()),
                        "objective_mean": np.max(self.objective(torch.vstack(self.list_mu)).cpu().numpy()),
                        "current objective_mean": self.objective(self.list_mu[-1]).cpu().numpy(),
                        "Avg distance samples": avg_distance,
                        "Avg distance samples from center": avg_from_center,
                        "Ellipse volume": covar_volume}
        # elif self.objective.dim == 2:
        #     raise NotImplementedError
        #     fig, ax = plt.subplots()
        #     bounds = self.objective.bounds
        #     lb, up = float(bounds[0][0]), float(bounds[1][0])
        #     ax.set_xlim(lb, up)
        #     plot_gp_fit(ax, self.model, self.train_x, targets=self.train_y, obj=self.objective, batch=self.batch_size, normalize_flag=True)
        #     #fig.savefig(os.path.join(self.plot_path, f"synthesis_{self.iteration}.png"))
            
        #     buf = BytesIO()
        #     plt.savefig(buf, format='png')
        #     buf.seek(0)
        #     plt.close()

        #     # Open image with Pillow
        #     image_var = Image.open(buf)
        #     return image_var

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
        return {"objective": np.max(torch.vstack(self.values_history).cpu().numpy())}
        
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
        BETA, STD_PRIOR = optimizer_config["beta"], optimizer_config["std_prior"]**2
        self.beta = BETA
        mean, loc = optimizer_config["mean_prior"]*torch.ones(objective.dim, device=objective.device, dtype=objective.dtype), STD_PRIOR*torch.eye(objective.dim, device=objective.device, dtype=objective.dtype)
        self.distribution = MultivariateNormal(mean, loc)
        # self.distribution_normalized = normalize_distribution(self.distribution, self.objective.bounds)

        # Initialization of training data.
        # self.unit_cube = torch.tensor([[0.]*self.objective.dim, [1.]*self.objective.dim], dtype=self.objective.dtype, device=self.objective.device)
        self.train_x, self.train_y, _ = generate_data("piqEI", objective=objective, n_init=n_init, distribution=self.distribution)
        
        self.params_history_list = [self.train_x.clone()]
        self.values_history = [self.train_y.clone()]
        
        # Acquistion function and its optimization properties.
        self.qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([optimizer_config["mc_samples"]]))
        self.optimize_acqf = initialize_acqf_optimizer(
                                                bounds=self.objective.bounds,
                                                batch_size=self.batch_size,
                                                num_restarts=optimizer_config["num_restarts"],
                                                raw_samples=optimizer_config["raw_samples"],
                                                batch_acq=optimizer_config["batch_acq"],
                                                maxiter=200)
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
        self.model = SingleTaskGP(self.train_x,
                                    train_y_init_standardized,
                                    covar_module=covar_module).to(self.train_x)
        
        self.acquisition_function = piqExpectedImprovement(
                    model=self.model, 
                    best_f=self.model.train_targets.max(),
                    pi_distrib=self.distribution,
                    n_iter=self.iteration+1, ## here iteration starts at 0
                    beta=self.beta,
                    sampler=self.qmc_sampler
                )
        
        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)

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
                    pi_distrib=self.distribution,
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
        new_x = self.optimize_acqf(self.acquisition_function)
        # new_x_normalized = self.optimize_acqf(self.acquisition_function)
        # new_x = unnormalize(new_x_normalized, bounds=self.objective.bounds)
        new_y = self.objective(new_x).unsqueeze(-1)

        # Update training points.
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        self.params_history_list.append(new_x.clone())
        self.values_history.append(new_y.clone())
        self.iteration += 1
    
    def plot_synthesis(self):
        # if self.objective.dim == 1:
        #     fig, ax = plt.subplots()
        #     bounds = self.objective.bounds
        #     lb, up = float(bounds[0][0]), float(bounds[1][0])
        #     ax.set_xlim(lb, up)
        #     plot_gp_fit(ax, self.model, self.train_x, targets=self.train_y, obj=self.objective, batch=self.batch_size, normalize_flag=True)
        #     # fig.savefig(os.path.join(self.plot_path, f"synthesis_{self.iteration}.png"))
        #     buf = BytesIO()
        #     plt.savefig(buf, format='png')
        #     buf.seek(0)
        #     plt.close()

        #     # Open image with Pillow
        #     image = Image.open(buf)
        #     return image
        return {"objective": np.max(torch.vstack(self.values_history).cpu().numpy())}

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

## Warning: Be careful computational graphs when using .copy() as not deleted and can result in using too much memory
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
        ## Load parameter distribution
        mu, var = optimizer_config["mean_prior"], optimizer_config["std_prior"]**2
        mean, loc = mu*torch.ones(objective.dim, device=objective.device, dtype=objective.dtype), var*torch.eye(objective.dim, device=objective.device, dtype=objective.dtype)
        self.distribution = MultivariateNormal(mean, loc)

        self.train_x, self.train_y, _ = generate_data("quad", objective=objective, n_init = n_init, distribution=self.distribution)
        #self.params = self.train_x.clone()
        self.params_history_list = [self.train_x.clone()]
        # self.values_history = [self.train_y.clone()]
        self.values_history = [self.objective(self.distribution.loc).repeat(self.batch_size)]

        self.d = self.objective.dim
        
        self.aqc_type = optimizer_config["aqc_type"]
        self.policy=optimizer_config["policy"]
        self.gradient = optimizer_config["gradient"]
        self.type = optimizer_config["type"]
        self.chi_threshold = np.sqrt(scipy.stats.chi2.ppf(q=0.9973,df=self.d))
        self.mahalanobis = optimizer_config["mahalanobis"]
        self.t_update = optimizer_config["lr"]

        ## Create Manifold
        euclidean = geoopt.manifolds.Euclidean()
        self.manifold = geoopt.manifolds.ProductManifold((euclidean, objective.dim), (euclidean, (objective.dim,objective.dim)))

        ## Register mu and covariance history
        self.list_mu, self.list_covar = [self.distribution.loc.detach().clone()], [self.distribution.covariance_matrix.detach().clone()]

        ## Create grads points for autodiff
        mu, covar = self.distribution.loc, self.distribution.covariance_matrix
        self.manifold_point = geoopt.ManifoldTensor(torch.cat((mu, covar.flatten())), manifold=self.manifold)
        self.manifold_point.requires_grad = True
        self.param = geoopt.ManifoldParameter(self.manifold_point)
        
        # self.optimize_acqf = initialize_acqf_optimizer(type="random", candidate_vr=optimizer_config["candidates_vr"], batch_size=batch_size)
        self.optimize_acqf = initialize_acqf_optimizer(
                                                type = self.aqc_type,
                                                bounds=self.objective.bounds,
                                                batch_size=self.batch_size,
                                                num_restarts=optimizer_config["num_restarts"],
                                                raw_samples=optimizer_config["raw_samples"],
                                                batch_acq=optimizer_config["batch_acq"],
                                                candidates_vr=optimizer_config["candidates_vr"],
                                                maxiter=200)
        

        ### Make model and quad, only for plots in iteration 0
        self.create_model()

        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)
        self.quad_model = Quadrature(self.model, self.distribution)
        self.linesearch = load_linesearch(self.policy, quad_model = self.quad_model, dict_parameter=optimizer_config)

        ### Instead do gradient in reparameterization and same for natural gradient
        ## Compute direction (sampled, expected , mdp ect...).
        if self.gradient == "expected":
            m, _ = self.quad_model.quadrature(self.manifold.take_submanifold_value(self.manifold_point, 0), self.manifold.take_submanifold_value(self.manifold_point, 1))
            m.backward()
            self.d_manifold = self.manifold_point.grad.detach().clone()
            self.manifold_point.grad.zero_()
            self.d_mu, self.d_epsilon = self.manifold.take_submanifold_value(self.d_manifold, 0), self.manifold.take_submanifold_value(self.d_manifold, 1)
        elif self.gradient == "sampled":
            self.d_mu, self.d_epsilon = self.quad_model.gradient_estimate(10000)
            # distrib_gradient = self.quad_model.gradient_distribution()
            # self.gradient_joint = distrib_gradient.sample()
            # self.d_mu = self.gradient_joint[:self.d]
            # self.d_epsilon = self.gradient_joint[self.d:].reshape(self.d, self.d)
            # self.d_epsilon = 0.5*(self.d_epsilon + self.d_epsilon.T)
            
        
        # Taking natural gradient:
        ## TODO make it a function
        if self.type == "SNES":
            # Here it is assumed both matrix for gp kernel and input distribution are diagonal
            self.d_mu = torch.matmul(self.distribution.covariance_matrix, self.d_mu)
            self.d_epsilon = torch.matmul(torch.matmul(self.distribution.covariance_matrix, torch.diag(torch.diag(self.d_epsilon))), self.distribution.covariance_matrix)
        elif self.type == "XNES":
            A = torch.linalg.cholesky(self.distribution.covariance_matrix)
            self.d_mu = torch.matmul(self.distribution.covariance_matrix, self.d_mu)
            self.d_epsilon = torch.matmul(torch.matmul(A.T, self.d_epsilon), A)
        elif self.type == "CMAES":
            self.d_mu = torch.matmul(self.distribution.covariance_matrix, self.d_mu)
            self.d_epsilon = torch.matmul(torch.matmul(self.distribution.covariance_matrix, self.d_epsilon), self.distribution.covariance_matrix)
        else:
            raise NotImplementedError
        
        ## Line search mehod, decide how far the step take
        if self.policy != "constant":
            self.quad_model.precompute_linesearch(self.distribution.loc, self.distribution.covariance_matrix, [self.d_mu, self.d_epsilon])
        self.t_update = self.linesearch.compute_t()
    
    def step(self):
        """Make a gradient step toward"""
        
        # Taking gradient update:
        ## TODO make it a function
        if self.type == "SNES":
            # Here it is assumed both matrix for gp kernel and input distribution are diagonal
            mu_target = self.distribution.loc + self.t_update*self.d_mu
            covar_target = self.distribution.covariance_matrix * torch.exp(self.t_update*self.d_epsilon)
        elif self.type == "XNES":
            A = torch.linalg.cholesky(self.distribution.covariance_matrix)
            mu_target = self.distribution.loc + self.t_update*self.d_mu
            covar_target = torch.matmul(torch.matmul(A, torch.linalg.matrix_exp(self.t_update*self.d_epsilon)), A.T)
        elif self.type == "CMAES":
            self.manifold_point = self.manifold.expmap(self.manifold_point, self.t_update*torch.hstack([self.d_mu, self.d_epsilon.flatten()]))
            mu_target = self.distribution.loc + self.t_update*self.d_mu
            covar_target = self.distribution.covariance_matrix + self.t_update*self.d_epsilon
        else:
            raise NotImplementedError
        
        try:
            self.distribution = MultivariateNormal(mu_target, covar_target)
        except:
            print("Adding jitter for positive definiteness")
            covar_target  += (10e-6)*torch.eye(self.dim, device = covar_target.device, dtype=covar_target.dtype)
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

        self.params_history_list.append(new_x.clone())
        # self.values_history.append(new_y.clone())
        self.values_history.append(self.objective(self.distribution.loc).repeat(self.batch_size))
        self.iteration += 1

        ### Make model
        self.create_model()

        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)

        # Update quad and linesearch
        self.quad_model = Quadrature(self.model, self.distribution)
        self.linesearch.quad_model = self.quad_model

        ### Instead do gradient in reparameterization and same for natural gradient
        ## Compute direction (sampled, expected , mdp ect...).
        if self.gradient == "expected":
            m, _ = self.quad_model.quadrature(self.manifold.take_submanifold_value(self.manifold_point, 0), self.manifold.take_submanifold_value(self.manifold_point, 1))
            m.backward()
            self.d_manifold = self.manifold_point.grad.detach().clone()
            self.manifold_point.grad.zero_()
            self.d_mu, self.d_epsilon = self.manifold.take_submanifold_value(self.d_manifold, 0), self.manifold.take_submanifold_value(self.d_manifold, 1)
        elif self.gradient == "sampled":
            self.d_mu, self.d_epsilon = self.quad_model.gradient_estimate(10000)
            # distrib_gradient = self.quad_model.gradient_distribution()
            # self.gradient_joint = distrib_gradient.sample()
            # self.d_mu = self.gradient_joint[:self.d]
            # self.d_epsilon = self.gradient_joint[self.d:].reshape(self.d, self.d)
            # self.d_epsilon = 0.5*(self.d_epsilon + self.d_epsilon.T)
        # Taking natural gradient:
        ## TODO make it a function
        if self.type == "SNES":
            # Here it is assumed both matrix for gp kernel and input distribution are diagonal
            self.d_mu = torch.matmul(self.distribution.covariance_matrix, self.d_mu)
            self.d_epsilon = torch.matmul(torch.matmul(self.distribution.covariance_matrix, torch.diag(torch.diag(self.d_epsilon))), self.distribution.covariance_matrix)
        elif self.type == "XNES":
            self.d_mu = torch.matmul(self.distribution.covariance_matrix, self.d_mu)
            self.d_epsilon = torch.matmul(torch.matmul(A.T, self.d_epsilon), A)
        elif self.type == "CMAES":
            self.d_mu = torch.matmul(self.distribution.covariance_matrix, self.d_mu)
            self.d_epsilon = torch.matmul(torch.matmul(self.distribution.covariance_matrix, self.d_epsilon), self.distribution.covariance_matrix)
        else:
            raise NotImplementedError
        
        ## Line search mehod, decide how far the step take
        if self.policy != "constant":
            self.quad_model.precompute_linesearch(self.distribution.loc, self.distribution.covariance_matrix, [self.d_mu, self.d_epsilon])
        self.t_update = self.linesearch.compute_t()

    def create_model(self):
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=self.train_x.shape[-1],
                batch_shape=None,
                lengthscale_prior=GammaPrior(self.d, np.sqrt(self.d)),
            ),
            batch_shape=None,
            outputscale_prior=GammaPrior(6.0, 3.0),
        )
        if self.mahalanobis:
            mask = _is_in_ellipse(self.distribution.loc, self.distribution.covariance_matrix, self.train_x, self.chi_threshold)
            train_x = self.train_x[mask]
            train_y = self.train_y[mask]
        else:
            train_x = self.train_x
            train_y = self.train_y
        train_y_init_standardized = standardize(train_y)
        # train_y_standardized_mean = train_y.mean()
        # train_y_standardized_std = train_y.std()

        self.model = SingleTaskGP(train_x, train_y_init_standardized, covar_module=covar_module).to(train_x) ## Can input subset of the dataset

    def loss_ei(self):
        mu, Epsilon = self.manifold.take_submanifold_value(self.param, 0), self.manifold.take_submanifold_value(self.param, 1)
        mean, covar = self.compute_target_distribution_zero_order(mu, Epsilon)
        return -log_EI(mean, torch.sqrt(covar), self.best_f)

    def plot_synthesis(self, save_path=".", iteration=0):
        batch_samples = self.train_x[(-self.batch_size):]
        matrix_distances = torch.cdist(batch_samples, batch_samples)
        avg_distance = matrix_distances[torch.triu(torch.ones(matrix_distances.shape), diagonal=1) == 1].mean()
        avg_from_center = torch.linalg.norm((self.list_mu[-1] - self.params_history_list[-1]), dim=1).mean()
        covar_volume = torch.linalg.det(self.distribution.covariance_matrix)**(1/self.d)
        dict_plot = {"mean": self.distribution.loc,
                     "objective": np.max(torch.vstack(self.values_history).cpu().numpy()),
                     "objective_mean": np.max(self.objective(torch.vstack(self.list_mu)).cpu().numpy()),
                     "current objective_mean": self.objective(self.distribution.loc).cpu().numpy(),
                     "GP lengthscales norm":torch.linalg.norm(self.model.covar_module.base_kernel.lengthscale[0]),
                     "Avg distance samples": avg_distance,
                     "Avg distance samples from center": avg_from_center,
                     "Ellipse volume": covar_volume,
                     "lr": self.t_update}
        
        if self.objective.dim == 1:
            image = plot_synthesis_quad(self, iteration=iteration, save_path=self.plot_path, standardize=True)
            image = wandb.Image(image)
            dict_plot["Image synthesis info"] = image
            dict_plot["std"] = torch.sqrt(self.distribution.covariance_matrix)
            
        else:
            if self.policy != "constant":
                image = plot_synthesis_ls(self, iteration=iteration)
                image = wandb.Image(image)
                dict_plot["Image synthesis info"] = image
            if len(self.list_mu) > 1:
                dict_plot["difference mean"] = torch.linalg.norm(self.list_mu[-1] - self.list_mu[-2])
                dict_plot["difference covar"] = torch.linalg.norm(self.list_covar[-1] - self.list_covar[-2])
            else:
                dict_plot["difference mean"] = 0.
                dict_plot["difference covar"] = 0.
        return dict_plot

