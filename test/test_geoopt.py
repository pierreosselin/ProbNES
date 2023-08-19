import sys, os
# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import numpy as np
from typing import Callable, Optional, Tuple
from module.quadrature import Quadrature, QuadratureExploration, QuadratureExplorationBis
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import scipy.stats as stats
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
import geoopt



quad_distrib = MultivariateNormal(torch.tensor([0., 0.]), torch.diag(torch.tensor([1., 1.])))
train_X = torch.linspace(-3,3, 10).reshape(5,2)
train_Y = (-(train_X)**2 + 1.).sum(dim=1, keepdim=True)
covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=train_X.shape[-1],
                    batch_shape=None,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=None,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )

model = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

quad = Quadrature(model=model,
            distribution=quad_distrib)

## autodiff
mean_autodiff, covariance_autodiff = torch.tensor([1., 0.5], requires_grad=True), torch.diag(torch.tensor([0.2, 0.2], requires_grad=True))
m, v = quad._quadrature(mean_autodiff, covariance_autodiff)
m.backward()


## Analytic
quad.gradient_direction()

### Manifold spd gradients
euclidean = geoopt.manifolds.Euclidean()
spd = geoopt.manifolds.SymmetricPositiveDefinite()
manifold = geoopt.manifolds.ProductManifold((euclidean, 2), (spd, (2,2)))

t = torch.zeros(6)
t[0], t[1], t[2], t[5] = 1., 0.5, 0.2, 0.2
mani_tensor = geoopt.ManifoldTensor(t, manifold=manifold, requires_grad = True)
m, v = quad._quadrature(manifold.take_submanifold_value(mani_tensor, 0), manifold.take_submanifold_value(mani_tensor, 1))
m.backward()

### Manifold euclidean gradients
euclidean = geoopt.manifolds.Euclidean()
manifold = geoopt.manifolds.ProductManifold((euclidean, 2), (euclidean, (2,2)))

t = torch.zeros(6)
t[0], t[1], t[2], t[5] = 1., 0.5, 0.2, 0.2
mani_tensor_euclidean = geoopt.ManifoldTensor(t, manifold=manifold, requires_grad = True)
m, v = quad._quadrature(manifold.take_submanifold_value(mani_tensor_euclidean, 0), manifold.take_submanifold_value(mani_tensor_euclidean, 1))
m.backward()


print("Gradient autodiff pytorch:")
print("mu:", mean_autodiff.grad)
print("epsilon:", covariance_autodiff.grad)

print("Gradient analytic:")
print("mu:", quad.d_mu)
print("epsilon:", quad.d_epsilon)

print("Gradient autodiff manifold spd geoopt:")
print("mu:", manifold.take_submanifold_value(mani_tensor, 0).grad)
print("epsilon:", manifold.take_submanifold_value(mani_tensor, 1).grad)

print("Gradient autodiff pytorch:")
print("mu:", manifold.take_submanifold_value(mani_tensor, 0).grad)
print("epsilon:", manifold.take_submanifold_value(mani_tensor, 1).grad)