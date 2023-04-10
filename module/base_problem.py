from .objective import Objective, get_objective
from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from botorch.utils.transforms import standardize, normalize, unnormalize
import gpytorch
# TODO Issue two Problem classes, unify them maybe
# TODO Look into this Bound issue, code organisation

class Problem:
    def __init__(self,
                 objective: Objective,
                 dim:Optional[int] = 2,
                 device: torch.device = None,
                 dtype: torch.dtype = None,):
    
        self.objective = objective
        self.dim = dim
        self.device = device
        self.dtype = dtype

    def generate_initial_data(self,
                              n: Optional[int]=10,
                              ):
        # generate training data
        train_x = unnormalize(torch.rand(n, self.dim, device=self.device, dtype=self.dtype), self.bounds) ### Change initializer normal or discrete
        train_obj = self.objective(train_x).unsqueeze(-1)  # add output dimension
        best_observed_value = train_obj.max().item()
        return train_x, train_obj, best_observed_value

def get_problem(
        label: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        problem_kwargs: Optional[Dict[str, Any]] = None
) -> Problem:
    problem_kwargs = problem_kwargs or {}
    obj = get_objective(label=label, **problem_kwargs)
    bounds = problem_kwargs.get("initial_bounds", 10.)
    dim = problem_kwargs.get("dim", 2)
    pb = Problem(objective = obj, dim=dim, device=device, dtype=dtype)
    pb.bounds = torch.tensor([[-bounds] * pb.dim, [bounds] * pb.dim], device=device, dtype=dtype)
    return pb
        

