from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from botorch.models import SingleTaskGP
from module.quadrature import Quadrature
from tqdm import tqdm

objective = lambda x: -x**2
mean_distrib, var_distrib = torch.tensor([0.]), torch.diag(torch.tensor([1.]))
quad_distrib = MultivariateNormal(mean_distrib, var_distrib)
NORMALIZE = True

n = 10
bounds = 2
lb, up= -10., 10.
train_X = torch.linspace(-bounds,bounds, n, dtype=torch.float64).reshape(-1,1)
train_Y = (objective(train_X)).sum(dim=1, keepdim=True)

def posterior_quad(theta, var):
    mean_distrib_test, var_distrib_test = torch.tensor([theta], dtype=torch.float64), torch.diag(torch.tensor([var], dtype=torch.float64))
    quad_distrib_test = MultivariateNormal(mean_distrib_test, var_distrib_test)
    quad_test = Quadrature(model=model,
            distribution=quad_distrib_test,
            c1 = 0.1,
            c2 = 0.2,
            t_max = 1,
            budget = 50,
            maximize = True)
    
    quad_test.quadrature()
    return quad_test.m, quad_test.v

from scipy.stats import norm
def expected_improvement(theta, var, y_max, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    if var == 0.0:
            return 0.0
    
    with np.errstate(divide='warn'):
        imp = theta - y_max - xi
        Z = imp / var
        ei = imp * norm.cdf(Z) + var * norm.pdf(Z)
    return ei

if NORMALIZE:
    mean_Y = torch.mean(train_Y, dim = 0)
    std_Y = torch.std(train_Y, dim = 0)
    train_Y = (train_Y - mean_Y)/std_Y
else:
    mean_Y = None
    std_Y =None

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
            distribution=quad_distrib,
            c1 = 0.1,
            c2 = 0.2,
            t_max = 1,
            budget = 50,
            maximize = True)
quad.quadrature()
quad.gradient_direction()

b = np.arange(-15, 15, 0.5)
d = np.arange(0, 5, 0.1)[1:]**2
B, D = np.meshgrid(b, d)
n, m = b.shape[0], d.shape[0]
res = torch.stack((torch.tensor(B.flatten()), torch.tensor(D.flatten())), axis = 1).numpy()
result, result_ei, result_wolfe = [], [], []
result_ei = []
for el in tqdm(res):
    post = posterior_quad(el[0], el[1])
    result.append(post)
    result_ei.append(expected_improvement(post[0].detach().numpy(), np.sqrt(post[1].detach().numpy()), torch.max(train_Y).numpy()))
    
mean, var = torch.tensor(result).numpy()[:,0].reshape(m,n), torch.tensor(result).numpy()[:,1].reshape(m,n)
ei = torch.tensor(result_ei).numpy().reshape(m,n)


t_linspace = torch.linspace(0.01, 1., 200, dtype=train_X.dtype)
result_wolfe = []
for t in t_linspace:
    result_wolfe.append(quad.compute_p_wolfe(t))
wolfe_tensor = torch.tensor(result_wolfe)