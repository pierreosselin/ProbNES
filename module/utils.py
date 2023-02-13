import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, Any, Union, Tuple, Callable, Dict, List

class Sphere(SyntheticTestFunction):
    r"""Sphere test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = x^2

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        self._bounds = [(-10., 10.) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        part1 = torch.norm(X, dim=-1)**2
        return part1