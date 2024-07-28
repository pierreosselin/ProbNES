import numpy as np
from .utils import bounded_bivariate_normal_integral, nearestPD, isPD, EI, log_EI
import torch

from botorch.utils.probability.utils import (
    ndtr as Phi
)
"""Code for applying linesearch methods"""


def load_linesearch(label, quad_model=None, dict_parameter=None):
    if label == "constant":
        linesearch = Constant(dict_parameter["lr"])
    elif label == "wolfe":
        linesearch = Wolfe(quad_model = quad_model, tmin = dict_parameter["t_min"], tmax=dict_parameter["t_max"], budget = dict_parameter["budget"], c1 = dict_parameter["c1"], c2 = dict_parameter["c2"])
    elif label == "armijo":
        raise NotImplementedError
    elif label == "EI":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return linesearch

class Linesearch:
    def __init__(self) -> None:
        pass

    def compute_t(self) -> float:
        pass

class Constant(Linesearch):
    def __init__(self, t) -> None:
        super(Constant, self).__init__()
        self.t = t
    
    def compute_t(self) -> float:
        return self.t

class Wolfe(Linesearch):
    def __init__(self, quad_model, tmin = 0.5, tmax=2., budget = 20, c1 = 0.0001, c2 = 0.9) -> None:
        super(Wolfe, self).__init__()
        self.quad_model = quad_model
        self.tmin = tmin
        self.tmax = tmax
        self.budget = budget
        self.c1 = c1
        self.c2 = c2

        assert 0 <= self.c1
        assert self.c1 < self.c2
        assert self.c2 <= 1

    def compute_t(self) -> float:
        current_max = -np.inf
        current_tmax = 1
        for t in np.linspace(self.tmin, self.tmax, self.budget):
            if self.compute_wolfe(t) > current_max:
                current_tmax = t
        return current_tmax

    def compute_wolfe(self, t):
        target_mu, target_epsilon = self.quad_model.mu1 + t*self.quad_model.d_mu, self.quad_model.Epsilon1 + t*self.quad_model.d_epsilon
        mean_joint, covar_joint = self.quad_model.jointdistribution_linesearch(target_mu, target_epsilon)
        
        if mean_joint == None and covar_joint == None:
            return 0
        
        return float(self.compute_wolfe_from_joint(-mean_joint, covar_joint, t)) # -mean_joint because we maximize

    def compute_wolfe_from_joint(self, mean_joint, covar_joint, t):
        A_transform = torch.tensor([[1, self.c1*t, -1, 0], [0, -self.c2, 0, 1]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device)
        if not isPD(covar_joint):
            return torch.tensor(0.)
        
        mean_wolfe = A_transform @ mean_joint
        covar_wolfe = A_transform @ covar_joint @ A_transform.T
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        rho = (covar_wolfe[0,1]/torch.sqrt(covar_wolfe[0,0]*covar_wolfe[1,1])).detach().cpu()
        al = -(mean_wolfe[0]/torch.sqrt(covar_wolfe[0,0])).detach().cpu()
        bl = -(mean_wolfe[1]/torch.sqrt(covar_wolfe[1,1])).detach().cpu()
        
        return bounded_bivariate_normal_integral(rho, al, torch.inf, bl, torch.inf)
  

def loss_ei(self):
    mu, Epsilon = self.manifold.take_submanifold_value(self.param, 0), self.manifold.take_submanifold_value(self.param, 1)
    mean, covar = self.compute_target_distribution_zero_order(mu, Epsilon)
    return -log_EI(mean, torch.sqrt(covar), self.best_f)

def compute_armijo(self, mean_joint, covar_joint, t):
    A_transform = torch.tensor([[1, self.c1*t, -1]], dtype=self.train_X.dtype, device=self.train_X.device)
    if not isPD(covar_joint):
        return torch.tensor(0.)
    
    mean_armijo = (A_transform @ mean_joint).squeeze(0)
    sigma_armijo = torch.sqrt(A_transform @ covar_joint @ A_transform.T).squeeze(0).squeeze(0)
    # Very small variances can cause numerical problems. Safeguard against
    # this with a deterministic evaluation of the Wolfe conditions.
    
    # Compute correlation factor and integration bounds for adjusted p_Wolfe
    # and return the result of the bivariate normal integral.
    
    return Phi(mean_armijo/sigma_armijo)

def armijo_criterion(self, t):
    target_point = self.manifold.expmap(self.manifold_point, t*self.d_manifold)
    mean_joint, covar_joint = self.compute_joint_distribution_first_order(self.distribution.loc, self.distribution.covariance_matrix, self.manifold.take_submanifold_value(target_point, 0), self.manifold.take_submanifold_value(target_point, 1))
    mean_joint = -mean_joint
    return -float(self.compute_armijo(mean_joint, covar_joint, t).detach().clone().cpu())