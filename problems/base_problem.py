## Generate problem

# Base Problem class
from typing import Optional
import torch

class Objective:

    def __init__(self,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 log: bool = False):
        self.noise_std = noise_std
        self.negate = negate
        self.log = log
        self.problem_size = None

    @property
    def is_moo(self) -> bool:
        raise NotImplementedError

    def evaluate_true(self, X):
        """True Objective functions"""
        raise NotImplementedError

    def __call__(self, X: torch.Tensor, noise: bool = False):
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X).to(dtype=torch.float, device=X.device)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        if self.log:
            f = torch.log(f)
        f += 1e-6 * torch.randn_like(f)
        return f if batch else f.squeeze(0)
