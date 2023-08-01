import numpy as np
from probnum.quad.integration_measures import IntegrationMeasure, GaussianMeasure

from probnum.quad.solvers import BayesianQuadrature, BQIterInfo
from probnum.quad.typing import DomainLike
from probnum.quad._bayesquad import _check_domain_measure_compatibility
from probnum.quad import bayesquad_from_data
from botorch.utils.transforms import standardize
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from typing import Callable, Optional, Tuple
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def bayesquad_from_initial_data(
    fun: Callable,
    nodes: np.ndarray,
    fun_evals: np.ndarray,
    kernel: Optional[Kernel] = None,
    measure: Optional[IntegrationMeasure] = None,
    domain: Optional[DomainLike] = None,
    policy: Optional[str] = "bmc",
    rng: Optional[np.random.Generator] = None,
    options: Optional[dict] = None
) -> Tuple[Normal, BQIterInfo]:
    r"""Perform Bayesian Quadraturefrom a given set of nodes and function
    evaluations.
    Parameters
    ----------
    nodes
        *shape=(n_eval, input_dim)* -- Locations at which the function evaluations are
        available as ``fun_evals``.
    fun_evals
        *shape=(n_eval,)* -- Function evaluations at ``nodes``.
    kernel
        The kernel used for the GP model. Defaults to the ``ExpQuad`` kernel.
    measure
        The integration measure. Defaults to the Lebesgue measure.
    domain
        The integration domain. Contains lower and upper bound as scalar or
        ``np.ndarray``. Obsolete if ``measure`` is given.
    options
        A dictionary with the following optional solver settings
            scale_estimation : Optional[str]
                Estimation method to use to compute the scale parameter. Defaults
                to 'mle'. Options are
                ==============================  =======
                 Maximum likelihood estimation  ``mle``
                ==============================  =======
            jitter : Optional[FloatLike]
                Non-negative jitter to numerically stabilise kernel matrix
                inversion. Defaults to 1e-8.
    Returns
    -------
    integral :
        The integral belief subject to the provided measure or domain.
    info :
        Information on the performance of the method.
    Raises
    ------
    ValueError
        If neither a domain nor a measure are given.
    Warns
    -----
    UserWarning
        When ``domain`` is given but not used.
    See Also
    --------
    bayesquad : Computes the integral using an acquisition policy.
    Warnings
    --------
    Currently the method does not support tuning of the kernel parameters
    other than the global kernel scale. Hence, the method may perform poorly unless the
    kernel parameters are set to appropriate values by the user.
    """

    if nodes.ndim != 2:
        raise ValueError(
            "The nodes must be given a in an array with shape=(n_eval, input_dim)"
        )

    input_dim, domain, measure = _check_domain_measure_compatibility(
        input_dim=nodes.shape[1], domain=domain, measure=measure
    )

    bq_method = BayesianQuadrature.from_problem(
        input_dim=input_dim,
        kernel=kernel,
        measure=measure,
        domain=domain,
        policy=policy,
        initial_design=None,
        options=options,
    )

    # Integrate
    integral_belief, bqstate, info = bq_method.integrate(
        fun=fun, nodes=nodes, fun_evals=fun_evals, rng=rng
    )

    return integral_belief, bqstate, info

## Should be able to compute integrale and gradient with uncertainty
class Quadrature:
    def __init__(self, 
                 objective=None,
                 distribution=None,
                 batch_size=None,
                 device=None,
                 dtype=None,
                 policy=None,
                 params=None,
                 step_size=1.) -> None:
        self.objective=objective
        self.distribution=distribution
        self.batch_size=batch_size
        self.device=device
        self.dtype=dtype
        self.policy=policy
        self.kernel=None
        self.options={"max_evals":batch_size, "batch_size": batch_size}
        self.fun = lambda x: self.objective(torch.tensor(x, device=device, dtype=dtype)).cpu().detach().numpy()
        self.params=params
        self.step_size=step_size

    def update(self,
               train_x=None,
               train_y=None):
        self.train_x=train_x
        self.train_y=train_y
        self.nodes = self.train_x.cpu().detach().numpy()
        if self.nodes.ndim == 1:
            self.nodes = self.nodes.reshape(-1,1)
        self.fun_evals = train_y.flatten().cpu().detach().numpy()
        self.measure = GaussianMeasure(self.distribution.loc.cpu().detach().numpy(), self.distribution.covariance_matrix.cpu().detach().numpy())
        self.options["max_evals"] = self.train_x.shape[0] + self.batch_size

    def integrate(self):
        F, bqstate, _ = bayesquad_from_initial_data(fun=self.fun,nodes=self.nodes, fun_evals=self.fun_evals, measure=self.measure, policy=self.policy, options=self.options, rng=np.random.default_rng(0))
        new_x = torch.tensor(bqstate.nodes[-self.batch_size:], device=self.device, dtype=self.dtype)
        new_obj = torch.tensor(bqstate.fun_evals[-self.batch_size:], device=self.device, dtype=self.dtype).reshape(-1,1)
        return F, new_x, new_obj
    
    def grad_integration(self):
        ## Given dataset, update grad distribution
        ## Distribution agnostic but be careful params management + constraints
        ## Only tackle gaussian for now
        loc, cov = self.distribution.loc, self.distribution.covariance_matrix
        inv_cov = torch.linalg.inv(cov)
        inv_c = torch.diag(inv_cov)
        grad_mu = (self.train_x - loc) @ inv_cov.T
        grad_cov = 0.5*inv_c*(inv_c*((self.train_x - loc)**2) -1)
        grad_param = torch.concat((grad_mu, grad_cov), dim=1)
        grad_train_y = self.train_y*grad_param
        list_grad_mean, list_grad_var = [], []
        for el in grad_train_y.T:
            F, _ = bayesquad_from_data(nodes=self.nodes, fun_evals=el.flatten().cpu().detach().numpy(), measure=self.measure)
            list_grad_mean.append(float(F.mean))
            list_grad_var.append(float(F.var))
        self.grad_distribution = MultivariateNormal(torch.tensor(list_grad_mean, device=self.device, dtype=self.dtype), torch.diag(torch.tensor(list_grad_var, device=self.device, dtype=self.dtype)))
        return None


    def update_distribution(self):
        ## To change for other than gaussian
        sample = self.grad_distribution.sample()
        self.params[0] += self.step_size * sample[:(self.params[0].shape[0])]
        self.params[1] += self.step_size * sample[(self.params[1].shape[0]):]
        self.params[1] = torch.maximum(self.params[1], 1e-5*torch.ones_like(self.params[1]))
        self.distribution = MultivariateNormal(torch.tensor(self.params[0], device=self.device, dtype=self.dtype), torch.diag(torch.tensor(self.params[1], device=self.device, dtype=self.dtype)))
        return self.distribution