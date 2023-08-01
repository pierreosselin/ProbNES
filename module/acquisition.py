from botorch.acquisition.monte_carlo import qExpectedImprovement
from typing import Any, Optional, Union
from botorch.models.model import Model
from torch.distributions.distribution import Distribution
from torch import Tensor
import torch
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from torch import Tensor
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)

class piqExpectedImprovement(qExpectedImprovement):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        pi_distrib: Distribution,
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
    
