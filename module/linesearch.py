import numpy as np
from .utils import bounded_bivariate_normal_integral, nearestPD, isPD, EI, log_EI, EI_bivariate, log_EI_bivariate
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
        linesearch = Armijo(quad_model = quad_model, tmin = dict_parameter["t_min"], tmax=dict_parameter["t_max"], budget = dict_parameter["budget"], c1 = dict_parameter["c1"])
    elif label == "ei":
        linesearch = EI_Linesearch(quad_model = quad_model, tmin = dict_parameter["t_min"], tmax=dict_parameter["t_max"], budget = dict_parameter["budget"])
    elif label == "ei_bi":
        linesearch = EI_BI_Linesearch(quad_model = quad_model, tmin = dict_parameter["t_min"], tmax=dict_parameter["t_max"], budget = dict_parameter["budget"])
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
        current_max = 0.3
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
    
    def compute_armijo(self, t):
        target_mu, target_epsilon = self.quad_model.mu1 + t*self.quad_model.d_mu, self.quad_model.Epsilon1 + t*self.quad_model.d_epsilon
        mean_joint, covar_joint = self.quad_model.jointdistribution_linesearch(target_mu, target_epsilon)

        if mean_joint == None and covar_joint == None:
            return 0
        
        return float(self.compute_armijo_from_joint(-mean_joint, covar_joint, t)) # -mean_joint because we maximize

    def compute_wolfe_from_joint(self, mean_joint, covar_joint, t):
        A_transform = torch.tensor([[1, self.c1*t, -1, 0], [0, -self.c2*t, 0, 1]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device)
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
    
    def compute_armijo_from_joint(self, mean_joint, covar_joint, t):
        mean_joint, covar_joint = mean_joint[:3], covar_joint[:3, :3]
        A_transform = torch.tensor([[1, self.c1*t, -1]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device)
        if not isPD(covar_joint):
            return torch.tensor(0.)
        
        mean_armijo = (A_transform @ mean_joint).squeeze(0)
        sigma_armijo = torch.sqrt(A_transform @ covar_joint @ A_transform.T).squeeze(0).squeeze(0)
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        
        return Phi(mean_armijo/sigma_armijo)
    
    def compute_slope_from_joint(self, mean_joint, covar_joint, t):
        mean_joint, covar_joint = torch.tensor([mean_joint[1], mean_joint[3]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device), torch.tensor([[covar_joint[1, 1], covar_joint[1, 3]], [covar_joint[3, 1], covar_joint[3, 3]]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device)
        A_transform = torch.tensor([[-self.c2*t, 1]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device)
        if not isPD(covar_joint):
            return torch.tensor(0.)
            
        mean_slope = (A_transform @ mean_joint).squeeze(0)
        sigma_slope = torch.sqrt(A_transform @ covar_joint @ A_transform.T).squeeze(0).squeeze(0)
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        
        return Phi(mean_slope/sigma_slope)
  
    def loss_ei(self, mean_joint, covar_joint, _):
        best_f, mean, covar = mean_joint[0], mean_joint[2], covar_joint[2,2]
        return -log_EI(mean, torch.sqrt(covar), best_f)
    
    def ei(self, mean_joint, covar_joint, _):
        best_f, mean, covar = mean_joint[0], mean_joint[2], covar_joint[2,2]
        return EI(mean, torch.sqrt(covar), best_f)
    
    def ei_bivariate(self, mean_joint, covar_joint, _):
        mean_joint, covar_joint = torch.tensor([mean_joint[0], mean_joint[2]]), torch.tensor([[covar_joint[0, 0], covar_joint[0, 2]], [covar_joint[2, 0], covar_joint[2, 2]]])
        return EI_bivariate(mean_joint, covar_joint)

class Armijo(Linesearch):
    def __init__(self, quad_model, tmin = 0.5, tmax=2., budget = 20, c1 = 0.0001) -> None:
        super(Armijo, self).__init__()
        self.quad_model = quad_model
        self.tmin = tmin
        self.tmax = tmax
        self.budget = budget
        self.c1 = c1

        assert 0 <= self.c1
        
    def compute_t(self) -> float:
        current_threshold = 0.8
        current_tmax = 1
        lt = np.linspace(self.tmin, self.tmax, self.budget)
        iteration = 0
        while (iteration < self.budget) and (self.compute_armijo(lt[iteration]) > current_threshold):
            current_tmax = lt[iteration]
            iteration += 1
        return current_tmax
    
    def compute_armijo(self, t):
        target_mu, target_epsilon = self.quad_model.mu1 + t*self.quad_model.d_mu, self.quad_model.Epsilon1 + t*self.quad_model.d_epsilon
        mean_joint, covar_joint = self.quad_model.jointdistribution_linesearch(target_mu, target_epsilon)

        if mean_joint == None and covar_joint == None:
            return 0
        
        return float(self.compute_armijo_from_joint(-mean_joint, covar_joint, t)) # -mean_joint because we maximize
    
    def compute_armijo_from_joint(self, mean_joint, covar_joint, t):
        mean_joint, covar_joint = mean_joint[:3], covar_joint[:3, :3]
        A_transform = torch.tensor([[1, self.c1*t, -1]], dtype=self.quad_model.train_X.dtype, device=self.quad_model.train_X.device)
        if not isPD(covar_joint):
            return torch.tensor(0.)
        
        mean_armijo = (A_transform @ mean_joint).squeeze(0)
        sigma_armijo = torch.sqrt(A_transform @ covar_joint @ A_transform.T).squeeze(0).squeeze(0)
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        
        return Phi(mean_armijo/sigma_armijo)
    
class EI_Linesearch(Linesearch):
    def __init__(self, quad_model, tmin = 0.5, tmax=2., budget = 20) -> None:
        super(EI_Linesearch, self).__init__()
        self.quad_model = quad_model
        self.tmin = tmin
        self.tmax = tmax
        self.budget = budget
        
    def compute_t(self) -> float:
        current_max = 0.3
        current_tmax = 1
        for t in np.linspace(self.tmin, self.tmax, self.budget):
            if self.compute_ei(t) > current_max:
                current_tmax = t
        return current_tmax
    
    def compute_ei(self, t):
        target_mu, target_epsilon = self.quad_model.mu1 + t*self.quad_model.d_mu, self.quad_model.Epsilon1 + t*self.quad_model.d_epsilon
        mean_joint, covar_joint = self.quad_model.jointdistribution_linesearch(target_mu, target_epsilon)

        if mean_joint == None and covar_joint == None:
            return 0
        
        return float(EI(mean_joint[2], torch.sqrt(covar_joint[2,2]), mean_joint[0])) # -mean_joint because we maximize
    

class EI_BI_Linesearch(Linesearch):
    def __init__(self, quad_model, tmin = 0.5, tmax=2., budget = 20) -> None:
        super(EI_BI_Linesearch, self).__init__()
        self.quad_model = quad_model
        self.tmin = tmin
        self.tmax = tmax
        self.budget = budget
        
    def compute_t(self) -> float:
        current_max = 0.3
        current_tmax = 1
        for t in np.linspace(self.tmin, self.tmax, self.budget):
            if self.compute_ei_bi(t) > current_max:
                current_tmax = t
        return current_tmax
    
    def compute_ei_bi(self, t):
        target_mu, target_epsilon = self.quad_model.mu1 + t*self.quad_model.d_mu, self.quad_model.Epsilon1 + t*self.quad_model.d_epsilon
        mean_joint, covar_joint = self.quad_model.jointdistribution_linesearch(target_mu, target_epsilon)

        if mean_joint == None and covar_joint == None:
            return 0
        
        mean_joint, covar_joint = torch.tensor([mean_joint[0], mean_joint[2]]), torch.tensor([[covar_joint[0, 0], covar_joint[0, 2]], [covar_joint[2, 0], covar_joint[2, 2]]])
        return float(EI_bivariate(mean_joint, covar_joint))