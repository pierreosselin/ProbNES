## Generate problem

# Base Problem class
from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Rastrigin
import numpy as np
from sklearn.preprocessing import StandardScaler





class Objective:
    def __init__(self,
                 obj_func: Callable,
                 noise_std: Optional[float] = None,
                 best_value: Optional[float] = None,
                 negate: bool = False,
                 dim:int = 2):
        self.obj_func = obj_func
        self.noise_std = noise_std
        self.best_value = best_value
        self.negate = negate
        self.dim = dim

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

    def __call__(self, X: torch.Tensor, noise: bool = False):
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        return f
    
def get_objective(
        label: str,
        problem_kwargs: Optional[Dict[str, Any]] = None,
) -> Objective:
    problem_kwargs = problem_kwargs or {}
    if label == "test_function":
        test_function = problem_kwargs.get("function", "rosenbrock")
        dim = problem_kwargs.get("dim", 2)
        noise_std = problem_kwargs.get("noise_std", 0.)
        if test_function == "rosenbrock":
            obj = Objective(obj_func=Rosenbrock(dim), noise_std=noise_std, best_value=0., negate=True, dim=dim)
        elif test_function == "ackley":
            obj = Objective(obj_func=Ackley(dim), noise_std=noise_std, best_value=0., negate=True, dim=dim)
        elif test_function == "rastrigin":
            obj = Objective(obj_func=Rastrigin(dim), noise_std=noise_std, best_value=0., negate=True, dim=dim)
        else:
            raise NotImplementedError(f"Function {test_function} is not implemented")
    
    elif label == "airfoil":
        ## Load data
        data = np.loadtxt('data/airfoil_self_noise.dat')
        scale= StandardScaler()
        scaled_data = scale.fit_transform(data)

        target_value = problem_kwargs.get("target_value", 100.)
        scaled_target_value = (target_value - scale.mean_[-1])/(scale.scale_[-1])
        
        ## Create dictionary
        d = {}
        for row in scaled_data:
            d[tuple(row[:5])] = row[5]

        ## Create objective
        def objec(x):
            if x.ndim > 1:
                res = []
                for btc in x:
                    el = tuple(btc.detach().cpu().numpy())
                    res.append(abs(d[el] - target_value))
                res = torch.tensor([el]).to(x)
            else:
                el  = tuple(x.detach().cpu().numpy())
                res = abs(d[el] - target_value)
                res = torch.tensor([el]).to(x)
            return res
        obj = Objective(obj_func=objec, noise_std=0., best_value=target_value, dim = 5)

    else:
        raise NotImplementedError(f"Problem {label} is not implemented")
    return obj


class Problem:
    def __init__(self,
                 objective: Objective,
                 dim:int = 2,
                 data: Optional[np.Array] = None):
    
        self.objective = objective
        self.data = data
        self.dim = dim

    