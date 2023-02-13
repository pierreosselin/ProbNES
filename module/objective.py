## Generate objective

from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Rastrigin
from .utils import Sphere

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
            
        else:
            raise NotImplementedError(f"Function {test_function} is not implemented")
    
    elif label == "airfoil":
        ## Load data
        scaled_data = problem_kwargs.get("scaled_data")
        scaled_target_value = problem_kwargs.get("scaled_target_value", None)
        
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
                    res.append(-abs(d[el] - scaled_target_value))
                res = torch.tensor(res).to(x)
            else:
                el  = tuple(x.detach().cpu().numpy())
                res = -abs(d[el] - scaled_target_value)
                res = torch.tensor([res]).to(x)
            return res
        
        obj = Objective(obj_func=objec, noise_std=0., best_value=0.)
        return obj
            
        

    else:
        raise NotImplementedError(f"Problem {label} is not implemented")
    return obj