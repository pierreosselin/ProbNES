import sys, os
# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from module.quadrature import Quadrature
from botorch.models import SingleTaskGP
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior

objective = lambda x: 5*torch.exp(-2*(x - 1)**2) + 5*torch.exp(-2*(x + 1)**2)

mean_distrib, var_distrib = torch.tensor([0.5]), torch.diag(torch.tensor([5.]))
quad_distrib = MultivariateNormal(mean_distrib, var_distrib)
NORMALIZE = True

n = 10
bounds = 5
lb, up= -5., 5.
train_X = torch.linspace(-bounds,bounds, n, dtype=torch.float64).reshape(-1,1)
#train_Y = (objective(train_X)).sum(dim=1, keepdim=True)
train_Y = objective(train_X).reshape(-1,1)

if NORMALIZE:
    mean_Y = torch.mean(train_Y, dim = 0)
    std_Y = torch.std(train_Y, dim = 0)
    train_Y = (train_Y - mean_Y)/std_Y
else:
    mean_Y = None
    std_Y = None

# RBF kernel + training
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
quad.quadrature()

mu = Normal(0., 1.).sample(torch.tensor([3,1]))
Epsilon= torch.abs(Normal(0., 1.).sample(torch.tensor([3,1,1])))

quad.compute_joint_distribution_zero_order(mu, Epsilon)