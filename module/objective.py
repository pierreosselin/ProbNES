## Generate objective

from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Rastrigin
from .utils import Sphere
import gpytorch

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Objective:
    def __init__(self,
                 obj_func: Callable,
                 noise_std: Optional[float] = None,
                 best_value: Optional[float] = None,
                 negate: bool = False):
        self.obj_func = obj_func
        self.noise_std = noise_std
        self.best_value = best_value
        self.negate = negate

    @torch.no_grad()
    def evaluate_true(self, X):
        if isinstance(X, torch.Tensor):
            if self.negate:
                return -self.obj_func(X)
            else:
                return self.obj_func(X)
        else:
            raise TypeError("Only torch tensor are allowed")

    @property
    def ground_truth(self):
        return self.best_value

    def raw_to_scaled(self):
        raise NotImplementedError

    def scaled_to_raw(self):
        raise NotImplementedError

    def __call__(self, X: torch.Tensor, noise: bool = False):
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        return f
    
def get_objective(
        label: str,
        **problem_kwargs,
) -> Objective:
    problem_kwargs = problem_kwargs or {}
    if label == "test_function":
        test_function = problem_kwargs.get("function", "rosenbrock")
        dim = problem_kwargs.get("dim", 2)
        noise_std = problem_kwargs.get("noise_std", 0.)
        if test_function == "rosenbrock":
            obj = Objective(obj_func=Rosenbrock(dim), noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "ackley":
            obj = Objective(obj_func=Ackley(dim), noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "rastrigin":
            obj = Objective(obj_func=Rastrigin(dim), noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "sphere":
            obj = Objective(obj_func=Sphere(dim), noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "function_1":
            obj_function = lambda x: torch.sin(x - 4.) + torch.sin((10./3.)*(x - 4.))
            dim = 1
            obj = Objective(obj_func=obj_function, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "mountains":
            obj_function = lambda x: torch.flatten(5*torch.exp(-2*(x - 1)**2) + 5*torch.exp(-2*(x + 1)**2))
            obj = Objective(obj_func=obj_function, noise_std=noise_std, best_value=5., negate=False)

        else:
            raise NotImplementedError(f"Function {test_function} is not implemented")
    
    elif label == "airfoil":
        # Devices and dtype
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.double
        
        ## Load data
        train_x, train_y = torch.load('data/airfoil_scaled_train_x.pt').to(dtype=dtype, device=device), torch.load('data/airfoil_scaled_train_y.pt').to(dtype=dtype, device=device)
        state_dict = torch.load('data/airfoil_model_state.pth')
        n = train_x.shape[0]
        noises = torch.ones(n, dtype=dtype, device=device) * 1e-5
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
        model = ExactGPModel(train_x, train_y, likelihood=likelihood)  # Create a new GP model
        model.load_state_dict(state_dict)
        model = model.to(dtype=dtype, device=device)
        target_value = train_y.cpu().max()
        
        ## Create objective
        def objec(x):
            x = x.clone().detach()
            if x.ndim == 1:
                x = x.reshape(-1,1)
            model.eval()
            likelihood.eval()

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(x))
            return observed_pred.mean
        
        obj = Objective(obj_func=objec, noise_std=0., best_value=target_value)
        return obj

    else:
        raise NotImplementedError(f"Problem {label} is not implemented")
    return obj