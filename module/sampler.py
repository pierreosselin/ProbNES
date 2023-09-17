"""Sampler class, from distribution and data output"""
import torch
from torch.distributions import MultivariateNormal

from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.kernels import RBFKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

from probnum.randprocs.kernels import ExpQuad
from probnum.quad._bayesquad import bayesquad_from_initial_data
from probnum.quad.integration_measures import GaussianMeasure
import numpy as np

from module.BASQ._basq import BASQ

def get_sampler(strategy, **kwargs):
    if strategy == "random":
        return Random_Sampler(batch_size = kwargs["batch_size"], dim = kwargs["dim"])
    elif strategy == "basq":
        return Basq_Sampler(batch_size = kwargs["batch_size"], dim = kwargs["dim"], objective = kwargs["objective"])
    elif strategy in ["bmc", "us_rand", "us", "mi_rand", "mi", "ivr_rand", "ivr"]:
        return Probnum_Sampler(batch_size = kwargs["batch_size"], dim = kwargs["dim"], objective = kwargs["objective"], policy = strategy)

class Sampler:
    def __init__(self, batch_size, dim) -> None:
        self.batch_size = batch_size
        self.dim = dim
    
    def update_sampler(self, **kwargs):
        pass

    def sample_batch(self):
        ## Sample a batch for the evolutionary search algorithm, has to be of size self.dim x batch_size
        pass


class Random_Sampler(Sampler):
    def __init__(self, batch_size, dim) -> None:
        super().__init__(batch_size, dim)

    def sample_batch(self, xmean, C, **kwargs):
        return MultivariateNormal(loc=xmean, covariance_matrix=C).sample((self.batch_size,)).T

class Probnum_Sampler(Sampler):
    def __init__(self, batch_size, dim, **kwargs) -> None:
        super().__init__(batch_size, dim)
        if "policy" not in kwargs.keys():
            self.policy = "ivr"
        else:
            self.policy = kwargs["policy"]
        self.objective = kwargs["objective"]
        self.fun = lambda x: self.objective(torch.tensor(x, device=self.objective.device, dtype=self.objective.dtype)).cpu().detach().numpy()

        

    def update_sampler(self, train_x, train_y, **kwargs):

        ### right now no ard and output scale as probnum does not support this. We do not normalize input input
        covar_module = RBFKernel(
                    ard_num_dims=None,
                    batch_shape=None,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )
        train_y_init_standardized = standardize(train_y)
        model = SingleTaskGP(train_x,
                             train_y_init_standardized,
                             covar_module=covar_module).to(train_x)
        
        # Optionally optimize hyperparameters.
        mll = ExactMarginalLogLikelihood(
            model.likelihood, model
        )

        fit_gpytorch_mll(mll) #Train gp surrogate
        
        model.covar_module
        self.kernel = ExpQuad(input_shape=(self.dim,), lengthscales=float(model.covar_module.lengthscale))
        self.neval = train_x.shape[0]
        self.nodes = train_x.cpu().detach().numpy()
        if self.nodes.ndim == 1:
            self.nodes = self.nodes.reshape(-1,1)
        self.fun_evals = train_y.flatten().cpu().detach().numpy()
        
        self.measure = GaussianMeasure(mean = kwargs["xmean"].clone().cpu().numpy(), cov = kwargs["C"].clone().cpu().numpy())

    def sample_batch(self, xmean, C, **kwargs):
        measure = GaussianMeasure(mean = xmean.clone().cpu().numpy(), cov = C.clone().cpu().numpy())
        _, bqstate, _ = bayesquad_from_initial_data(fun=self.fun, nodes=self.nodes, fun_evals=self.fun_evals, kernel = self.kernel, measure=measure, policy=self.policy, rng=np.random.default_rng(0), options={"max_evals":self.neval + self.batch_size, "batch_size": self.batch_size})
        new_x = torch.tensor(bqstate.nodes[-self.batch_size:])
        return new_x.T
         
    
class Basq_Sampler(Sampler):
    def __init__(self, batch_size, dim, **kwargs) -> None:
        super().__init__(batch_size, dim)
        self.device = kwargs
        if "device" in kwargs.keys():
            self.device = kwargs["device"]
        if "dtype" in kwargs.keys():
            self.device = kwargs["dtype"]
        self.objective = kwargs["objective"]
    def update_sampler(self, train_x, train_y, **kwargs):
        self.train_x = train_x
        self.train_y = train_y
        
    def sample_batch(self, xmean, C, **kwargs):
        distribution = MultivariateNormal(loc=torch.tensor(xmean), covariance_matrix=C).sample((self.batch_size,))
        train_y_init_standardized = standardize(self.train_y)
        self.quad = BASQ(
            self.train_x,  # initial locations
            train_y_init_standardized,  # initial observations
            prior = distribution,  # Gaussian prior distribution
            true_likelihood=self.objective,  # true likelihood to be estimated
            device=self.device,  # cpu or cuda
            dtype=self.dtype
            )
        new_x, new_obj = self.quad.run_basq()
        return new_x, new_obj
    
### Todo Wrapped gp quadrature?