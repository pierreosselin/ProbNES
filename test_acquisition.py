import numpy as np
from module.quadrature import Quadrature, QuadratureExploration, QuadratureExplorationBis
from botorch.models import SingleTaskGP
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from module.utils import nearestPD

def plot_GP_fit(model, distribution, train_X, targets, obj, normalize=False, lb=-10., up=10., mean_Y=None, std_Y=None):
    """ Plot the figures corresponding to the Gaussian process fit
    """
    model.eval()
    model.likelihood.eval()
    test_x = torch.linspace(lb, up, 200, device=train_X.device, dtype=train_X.dtype)
    with torch.no_grad():
        # Make predictions
        predictions = model.likelihood(model(test_x))
        lower, upper = predictions.confidence_region()
    
    if normalize:
        predictions = predictions*float(std_Y) + float(mean_Y)
        lower, upper = lower*float(std_Y) + float(mean_Y), upper*float(std_Y) + float(mean_Y)
        targets = targets*float(std_Y) + float(mean_Y)
    value_ = (obj(test_x.unsqueeze(-1))).flatten()

    plt.scatter(train_X.cpu().numpy(), targets.cpu().numpy(), color='black', label='Training data')
    plt.plot(test_x.cpu().numpy(), predictions.mean.cpu().numpy(), color='blue', label='Predictive mean')
    plt.plot(test_x.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='lightblue', alpha=0.5, label='Confidence region')
    
    x = np.linspace(distribution.loc - 3*distribution.covariance_matrix, distribution.loc + 3*distribution.covariance_matrix, 100).flatten()
    y_lim = plt.gca().get_ylim()
    plt.plot(x, (y_lim[1] - y_lim[0])*stats.norm.pdf(x, distribution.loc, distribution.covariance_matrix).flatten(), "k")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()


objective = lambda x: (x + 5) * torch.sin(x + 5)

from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior

mean_distrib, var_distrib = torch.tensor([1.]), torch.diag(torch.tensor([1.]))
quad_distrib = MultivariateNormal(mean_distrib, var_distrib)
NORMALIZE = True

n = 5
bounds = 4
lb, up= -10., 10.
train_X = torch.linspace(-bounds,1., n, dtype=torch.float64).reshape(-1,1)
train_Y = (objective(train_X)).sum(dim=1, keepdim=True)

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

BATCH_SIZE = 1
N_BATCH = 15

# Algorithm setting
NUM_RESTARTS = 10
RAW_SAMPLES = 512
MC_SAMPLES = 256
ACQUISITION_BATCH_OPTIMIZATION = 3
NORMALIZE = False
STANDARDIZE_LABEL = True
VERBOSE = True

def optimize_acqf_and_get_observation(acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        if NORMALIZE:
            bounds=torch.stack(
                [
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, dtype=torch.float64),
                ])
        else:
            bounds=torch.tensor([[-10.], [10.]], dtype=torch.float64)

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": ACQUISITION_BATCH_OPTIMIZATION, "maxiter": 200},
        )
        # observe new values
        new_x = candidates
        exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
        new_obj = exact_obj
        return new_x, new_obj

quad = Quadrature(model=model,
            distribution=quad_distrib,
            c1 = 0.1,
            c2 = 0.2,
            t_max = 1,
            budget = 50)
quad.quadrature()
acq = QuadratureExplorationBis(model=model, distribution= quad_distrib)
new_x, new_obj = optimize_acqf_and_get_observation(acq)