from .objective import Objective, get_objective
from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
# TODO Issue two Problem classes, unify them maybe
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

    def generate_initial_data(self):
        raise NotImplementedError

class DataProblem(Problem):
    def __init__(self,
                 objective: Objective,
                 dim:Optional[int] = 5,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 raw_data: Optional[np.array] = None,
                 scale: Optional[StandardScaler] = None,
                 scaled_data: Optional[np.array] = None
                 ):
        super().__init__(objective=objective, dim=dim, device=device, dtype=dtype)
        self.raw_data = raw_data
        self.scale = scale
        self.scaled_data = scaled_data

    def generate_initial_data(self,
                              n: Optional[int]=10,
                              ):
        
        # generate training data
        n_data = self.raw_data.shape[0]
        indice = torch.tensor(random.sample(range(n_data), n))
        train_x = torch.tensor(self.scaled_data[indice, :self.dim], device=self.device, dtype=self.dtype) ### Change initializer normal or discrete
        train_obj = self.objective(train_x).unsqueeze(-1)  # add output dimension
        best_observed_value = train_obj.max().item()
        return train_x, train_obj, best_observed_value
    
class SyntheticProblem(Problem):
    def __init__(self,
                 objective: Objective,
                 dim:Optional[int] = 2,
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ):
        super().__init__(objective=objective, dim=dim, device=device, dtype=dtype)

    def generate_initial_data(self,
                              n: Optional[int]=10,
                              ):
        # generate training data
        train_x = 20*torch.rand(n, self.dim, device=self.device, dtype=self.dtype) - 10 ### Change initializer normal or discrete
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
    if label == "airfoil":
        ## Load data and scale
        data = np.loadtxt('data/airfoil_self_noise.dat')
        scale = StandardScaler()
        scaled_data = scale.fit_transform(data)
        target_value = problem_kwargs.get("target_value", data[:, 5].max())
        scaled_target_value = (target_value - scale.mean_[-1])/(scale.scale_[-1])

        ## Get Objective function
        problem_kwargs["scaled_data"] = scaled_data
        problem_kwargs["scaled_target_value"] = scaled_target_value
        obj = get_objective(label=label, **problem_kwargs)

        ## Get pb
        pb = DataProblem(objective = obj, dim=5, device=device, dtype=dtype, raw_data=data, scale=scale, scaled_data=scaled_data)
        pb.data_size = data.shape[0]

    elif label == "test_function":
        ## Get functions and objective
        dim = problem_kwargs.get("dim", 2)
        obj = get_objective(label=label, **problem_kwargs)

        ## Get pb
        bounds = problem_kwargs.get("initial_bounds", 10.)
        pb = SyntheticProblem(objective=obj, dim=dim, device=device, dtype=dtype)
        pb.bounds = torch.tensor([[-bounds] * pb.dim, [bounds] * pb.dim], device=device, dtype=dtype)

    return pb
        

