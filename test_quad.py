import numpy as np
from probnum.quad import bayesquad
from probnum.quad.integration_measures import IntegrationMeasure, GaussianMeasure
from probnum.quad.solvers import BayesianQuadrature, BQIterInfo
from probnum.quad.typing import DomainLike
from probnum.quad._bayesquad import _check_domain_measure_compatibility
import torch
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from typing import Callable, Optional, Tuple
from torch.distributions.multivariate_normal import MultivariateNormal
from module.quadrature import Quadrature

train_x = torch.linspace(0, 1, 5).reshape(-1,1)
train_y = (train_x**2).flatten()
nodes = np.array([[10.]])
fun_evals = nodes.reshape(-1, )

fun = lambda x: x**2
batch_size = 5



dtype = torch.float64
device = "cpu"
kernel = None
measure = GaussianMeasure(0., 1.)
domain = None  # Obsolete if measure is given
policy = "ivr_rand" #"mi" "ivr" "us" "us_rand", "mi_rand", "ivr_rand"
options = {"max_evals": 10, "batch_size": batch_size} # 



mean, loc = torch.zeros(1, dtype=dtype, device=device, requires_grad=True), torch.eye(1, dtype=dtype, device=device, requires_grad=True)
params = [mean, loc]
distribution = MultivariateNormal(loc=params[0], covariance_matrix=params[1])


qd = Quadrature(objective=fun,
        distribution=distribution,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        policy=policy,
        options=options,
        params=params)
qd.update(train_x, train_y, distribution=distribution)
qd.integrate()
qd.grad_integration()