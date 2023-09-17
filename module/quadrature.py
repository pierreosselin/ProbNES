from typing import Dict, Optional, Tuple, Union, Any
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform

from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from typing import Callable, Optional, Tuple
import torch
from scipy.optimize import minimize_scalar
from torch.distributions.multivariate_normal import MultivariateNormal
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from .utils import bounded_bivariate_normal_integral, nearestPD, isPD, EI, log_EI
import geoopt
from botorch.utils.probability.utils import (
    ndtr as Phi
)

""" 
Be careful as:
- Gaussian Process on Gpytorch is done with exponential and kernel scalar (no normalization constant for the density)
- Formula assume normal distribution and no theta => use formula with a constant outputscale * normalizing constant
"""
# TODO Remove best_f for wolfe as not necessary
# TODO Refactor for policy and joint probability computation
# TODO Make everything work for mixture of gaussian

class Quadrature:
    def __init__(self, 
                 model=None,
                 distribution=None,
                 policy = "wolfe", ## Here decide which policy to use to update distribution, either wolfe, ei or wolfe_ei 
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 manifold = True,
                 ) -> None:
        
        ### Create attributes
        self.model=model
        self.distribution=distribution
        self.d = self.distribution.loc.shape[0]
        self.policy=policy
        self.device = model.train_inputs[0].device
        self.dtype = model.train_inputs[0].dtype
        self.best_f = self.model.train_targets.max() ## Needs to be deleted for policy different than ei

        # Policy options
        policy_options = dict(c1=0.1, c2=0.9, t_max=100, budget=500, iteration=1000)
        policy_options.update(policy_kwargs or {})        
        for k, v in policy_options.items():
            setattr(self, k, v)
        
        # Do one computation
        assert 0 <= self.c1
        assert self.c1 < self.c2
        assert self.c2 <= 1

        ## Create Manifold
        if manifold:
            euclidean = geoopt.manifolds.Euclidean()
            spd = geoopt.manifolds.SymmetricPositiveDefinite()
            self.manifold = geoopt.manifolds.ProductManifold((euclidean, self.d), (spd, (self.d,self.d)))
        else:
            self.manifold = geoopt.manifolds.ProductManifold((euclidean, self.d), (euclidean, (self.d,self.d)))
        mu, covar = self.distribution.loc, self.distribution.covariance_matrix
        self.manifold_point = geoopt.ManifoldTensor(torch.cat((mu, covar.flatten())), manifold=self.manifold)
        self.manifold_point.requires_grad = True
        self.param = geoopt.ManifoldParameter(self.manifold_point)
            
        # Extract model quantities
        self.train_X = self.model.train_inputs[0]
        self.noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(self.train_X.shape[0], dtype=self.train_X.dtype, device=self.train_X.device)
        self.gp_covariance = ((torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone()))**2).detach().clone()
        self.outputscale = self.model.covar_module.outputscale.detach().clone()
        self.K_X_X = (self.model.covar_module(self.train_X) + self.noise_tensor).evaluate().detach().clone()
        self.mean_constant = self.model.mean_module.constant.detach().clone()
        self.inverse_data_covar_y = torch.linalg.solve(self.K_X_X, self.model.train_targets - self.mean_constant)
        self.inverse_data = torch.linalg.inv(self.K_X_X)

        ## Create criterion
        if self.policy == "ei": # Change for multivariate gaussian 
            self.optimizer = geoopt.optim.RiemannianAdam((self.param,), lr=1e-2)
        
        elif self.policy in ["wolfe", "armijo"]:
            m, _ = self._quadrature(self.manifold.take_submanifold_value(self.manifold_point, 0), self.manifold.take_submanifold_value(self.manifold_point, 1))
            m.backward()
            if policy == "wolfe":
                def criterion(t):
                    target_point = self.manifold.expmap(self.manifold_point, t*self.manifold_point.grad)
                    mean_joint, covar_joint = self.compute_joint_distribution(self.manifold.take_submanifold_value(target_point, 0), self.manifold.take_submanifold_value(target_point, 1))
                    mean_joint = -mean_joint
                    return -float(self.compute_wolfe(mean_joint, covar_joint, t))
            elif policy == "armijo":
                def criterion(t):
                    target_point = self.manifold.expmap(self.manifold_point, t*self.manifold_point.grad)
                    mean_joint, covar_joint = self.compute_joint_distribution_first_order(self.manifold.take_submanifold_value(target_point, 0), self.manifold.take_submanifold_value(target_point, 1))
                    mean_joint = -mean_joint
                    return -float(self.compute_armijo(mean_joint, covar_joint, t).detach().clone().cpu())
            self.criterion = criterion

        #### Pre computation quantities for quadrature, gradient and method of lines
        # Constant necessary to use the formulae applicable for normal distribution
        self.covariance_gp_distr = self.gp_covariance + self.distribution.covariance_matrix
        self.constant = (self.model.covar_module.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()
        self.t_1X = self.constant * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.covariance_gp_distr).log_prob(self.train_X))
        self.R_11 = self.constant * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.covariance_gp_distr + self.distribution.covariance_matrix).log_prob(self.distribution.loc))
        self.inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, self.t_1X) #Independant of t!..
        self.mean_1 = self.mean_constant + (self.t_1X.T @ self.inverse_data_covar_y)
        self.var_1 = self.R_11 - self.t_1X.T @ self.inverse_data_covar_t1X

        ## For higher order in derivation
        Pi_1_1 = self.gp_covariance + 2*self.distribution.covariance_matrix
        self.Pi_inv_1_1 = torch.linalg.inv(Pi_1_1)
        self.fourth_order_Pi1_Pi1 = torch.einsum("ij,kl->ijkl", self.Pi_inv_1_1, self.Pi_inv_1_1)

        # Compute tau matrix
        self.diff_data = (self.train_X - self.distribution.loc) # Nxd
        self.normal_data_unsqueezed = torch.unsqueeze(self.t_1X, -1).repeat(1,self.d)
        self.Tau_mu1 = torch.linalg.solve(self.covariance_gp_distr, self.diff_data * self.normal_data_unsqueezed, left=False)
        
        self.normal_data_unsqueezed = torch.unsqueeze(self.normal_data_unsqueezed, -1).repeat(1,1,self.d)
        self.outer_prod = torch.bmm(torch.unsqueeze(self.diff_data, -1), torch.unsqueeze(self.diff_data, 1))
        self.outer_prod = (self.outer_prod - self.covariance_gp_distr) * self.normal_data_unsqueezed
        self.Tau_epsilon1 = 0.5 * torch.linalg.solve(self.covariance_gp_distr, torch.linalg.solve(self.covariance_gp_distr, self.outer_prod, left=False))

    # The rbf kernel is not a Gaussian kernel (a multiplicative constant is missing)
    def _quadrature(self, mean, covariance):
        ### quadrature function for optimization and gradient differentiation
        covariance_gp_distr = self.gp_covariance + covariance
        constant = (self.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()
        
        t_1X = constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = covariance_gp_distr).log_prob(self.train_X))
        R_11 = constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = covariance_gp_distr + covariance).log_prob(mean))

        inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, t_1X) #Independant of t!..
        mean_1 = self.mean_constant + (t_1X.T @ self.inverse_data_covar_y)
        var_1 = R_11 - t_1X.T @ inverse_data_covar_t1X
        
        return mean_1, var_1

    # The rbf kernel is not a Gaussian kernel (a multiplicative constant is missing)
    def quadrature(self):
        self.m = self.mean_1
        self.v = self.var_1

    def loss_ei(self):
        mu, Epsilon = self.manifold.take_submanifold_value(self.param, 0), self.manifold.take_submanifold_value(self.param, 1)
        mean, covar = self.compute_target_distribution_zero_order(mu, Epsilon)
        return -log_EI(mean, torch.sqrt(covar), self.best_f)

    def gradient_direction(self, sample = False):
        ## Check output scale
        self.gradient_m_mu = self.Tau_mu1.T @ self.inverse_data_covar_y
        self.gradient_m_epsilon = self.Tau_epsilon1.T @ self.inverse_data_covar_y

        ### Sample a gradient or take the expected gradient direction
        if sample:
            print("Sampling gradient not yet implemented properly")
            self.inverse_data_covar_t = self.inverse_data_covar_t1X
            self.R_prime = -0.5 * self.theta * torch.linalg.inv(self.covariance_gp_distr + self.distribution.covariance_matrix) * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.covariance_gp_distr).log_prob(self.distribution.loc))
            self.gradient_v_mu = self.Tau_mu1.T @ self.inverse_data_covar_t
            self.gradient_v_epsilon = self.R_prime - self.Tau_epsilon.T @ self.inverse_data_covar_t
            s = torch.normal(0, 1, size=(1,), device = self.train_X.device, dtype=self.train_X.dtype)
            self.d_mu = self.theta*(self.gradient_m_mu + s*self.gradient_v_mu)
            self.d_epsilon = self.theta*(self.gradient_m_epsilon + s*self.gradient_v_epsilon)
        else:
            self.d_mu = self.gradient_m_mu
            self.d_epsilon = self.gradient_m_epsilon
    
        ## Precompute quantities for method of lines
        self.R11_prime_epsilon = -0.5*self.R_11*self.Pi_inv_1_1
        self.R1_prime_1_prime_mu_mu = self.Pi_inv_1_1 * self.R_11
        self.R1_prime_1_prime_epsilon_epsilon = (self.fourth_order_Pi1_Pi1 + torch.einsum("ijkl->ikjl" ,self.fourth_order_Pi1_Pi1) + torch.einsum("ijkl->iljk" ,self.fourth_order_Pi1_Pi1)) * self.R_11

        self.mean_prime_1 = (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_y)).sum()
        self.cov_1_1_prime = (self.d_epsilon*self.R11_prime_epsilon).sum() - (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_t1X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_t1X)).sum()
        self.var_prime_1 = self.d_mu @ self.R1_prime_1_prime_mu_mu @ self.d_mu \
                    + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * self.R1_prime_1_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()

    
    def update_distribution(self):
        if self.policy in ["wolfe", "armijo"]:
            self.t_update = minimize_scalar(self.criterion).x
            updated_manifold_point = self.manifold.expmap(self.manifold_point, self.t_update*self.manifold_point.grad)
            mu_target, covar_target = self.manifold.take_submanifold_value(updated_manifold_point, 0), self.manifold.take_submanifold_value(updated_manifold_point, 1)
            distribution = MultivariateNormal(mu_target, covar_target)
            return distribution

        ## TODO Modufy for mixture of Gaussian
        ## TODO Check for gradient 
        elif self.policy == "ei":
            for _ in range(self.iteration):
                self.optimizer.zero_grad()
                output = self.loss_ei()
                output.backward()
                self.optimizer.step()
            mu_target, covar_target = self.manifold.take_submanifold_value(self.param, 0), self.manifold.take_submanifold_value(self.param, 1)
            return MultivariateNormal(mu_target, covar_target)
 
    def maximize_step(self):

        if self.policy in ["wolfe", "armijo"]:
            self.t_update = minimize_scalar(self.criterion).x
            updated_manifold_point = self.manifold.expmap(self.manifold_point, self.t_update*self.manifold_point.grad)
            mu_target, covar_target = self.manifold.take_submanifold_value(updated_manifold_point, 0), self.manifold.take_submanifold_value(updated_manifold_point, 1)
            distribution = MultivariateNormal(mu_target, covar_target)
            return distribution

        ## TODO Modufy for mixture of Gaussian
        ## TODO Check for gradient 
        elif self.policy == "ei":
            for _ in range(self.iteration):
                self.optimizer.zero_grad()
                output = self.loss_ei()
                output.backward()
                self.optimizer.step()
            mu_target, covar_target = self.manifold.take_submanifold_value(self.param, 0), self.manifold.take_submanifold_value(self.param, 1)
            return MultivariateNormal(mu_target, covar_target)
        
        if self.policy == "wolfe":
            criterion = self.compute_p_wolfe
        elif self.policy == "armijo":
            criterion = self.compute_p_armijo
        t_linspace = torch.linspace(0, self.t_max, self.budget + 1, dtype=self.train_X.dtype, device=self.train_X.device)[1:]
        list_max = []
        for t in t_linspace:
            list_max.append(criterion(t))
        t_tensor = torch.tensor(list_max)
        self.t_update = t_linspace[torch.argmax(t_tensor)]

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

    def compute_wolfe(self, mean_joint, covar_joint, t):
        A_transform = torch.tensor([[1, self.c1*t, -1, 0], [0, -self.c2, 0, 1]], dtype=self.train_X.dtype, device=self.train_X.device)
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
    
    def compute_joint_distribution_zero_order(self, mu2, Epsilon2):
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # For now assume mu2 and epsilon2 are of shape (b1xb2x....xbk)xd and (b1xb2x....xbk)xdxd (assume no batching)
        # Compute mu and PI

        mu1 = self.distribution.loc
        Epsilon1 = self.distribution.covariance_matrix
        
        if not isPD(Epsilon2):
            return None
        
        Pi_1_2 = self.gp_covariance + Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2


        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))

        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + t2X @ self.inverse_data_covar_y

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - self.t_1X @ inverse_data_covar_t2X
        mean_joint = torch.cat((self.mean_1.unsqueeze(0), mean_2.unsqueeze(0)))
        covar_joint = torch.cat((self.var_1.unsqueeze(0), cov_1_2.unsqueeze(0), cov_1_2.unsqueeze(0), var_2.unsqueeze(0))).reshape(2,2)
        return mean_joint, covar_joint
    
    def compute_target_distribution_zero_order(self, mu, Epsilon):
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # For now assume mu2 and epsilon2 are of shape (b1xb2x....xbk)xd and (b1xb2x....xbk)xdxd (assume no batching)
        # Compute mu and PI
        if not isPD(Epsilon):
            return None
        
        Pi_2_2 = self.gp_covariance + 2*Epsilon


        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu, covariance_matrix = Pi_2_2).log_prob(mu))
        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu, covariance_matrix=Epsilon + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon - self.gp_covariance) * normal_data_unsqueezed2
        
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.mean_constant + t2X @ self.inverse_data_covar_y

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        return mean_2, var_2
    
    def compute_joint_distribution_first_order(self, mu2, Epsilon2):
        """Computes joint distribution (f(0), f'(0), f(t))"""
        # Already changed dCov and Covd here
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        mu1, Epsilon1 = self.distribution.loc, self.distribution.covariance_matrix

        if not isPD(Epsilon2):
            Epsilon2 = nearestPD(Epsilon2)
            return None
        
        Pi_1_2 = self.gp_covariance + Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2
        Pi_inv_1_2 = torch.linalg.inv(Pi_1_2)
        nu = Pi_inv_1_2 @ (mu1 - mu2)
        
        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))
        R12_prime_mu = nu*R12
        R12_prime_epsilon = -0.5*(torch.linalg.inv(Pi_1_2) - nu @ nu.T)*R12
        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + t2X @ self.inverse_data_covar_y

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - self.t_1X @ inverse_data_covar_t2X
        
        product_R12_prime_mu, product_R12_prime_epsilon = (self.d_mu*R12_prime_mu).sum(), (self.d_epsilon*R12_prime_epsilon).sum()
        #R21_prime_mu = - R12_prime_mu and R21_prime_epsilon = R12_prime_epsilon
        cov_2_1_prime = -product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (self.Tau_mu1.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ inverse_data_covar_t2X)).sum()
        
        
        mean_joint = torch.tensor([self.mean_1, self.mean_prime_1, mean_2], dtype=self.train_X.dtype, device=self.train_X.device)
        covar_joint = torch.tensor([[self.var_1, self.cov_1_1_prime, cov_1_2],
                                    [self.cov_1_1_prime, self.var_prime_1, cov_2_1_prime],
                                    [cov_1_2, cov_2_1_prime, var_2]], dtype=self.train_X.dtype, device=self.train_X.device)
        return mean_joint.detach(), covar_joint.detach()

    def compute_joint_distribution(self, mu2, Epsilon2):
        # Already changed dCov and Covd here
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # Compute mu and PI
        mu1, Epsilon1 = self.distribution.loc, self.distribution.covariance_matrix
        
        if not isPD(Epsilon2):
            Epsilon2 = nearestPD(Epsilon2)
            return None
        
        Pi_1_2 = self.gp_covariance + Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2
        Pi_inv_1_2 = torch.linalg.inv(Pi_1_2)
        Pi_inv_2_2 = torch.linalg.inv(Pi_2_2)
        
        nu = Pi_inv_1_2 @ (mu1 - mu2)
        
        third_order_Pi_nu = torch.einsum("i,jk->ijk", nu, Pi_inv_1_2)
        third_order_nu_nu_nu = torch.einsum("i,j,k->ijk", nu, nu, nu)

        fourth_order_Pi_nu_nu = torch.einsum("ij,k,l->ijkl", Pi_inv_1_2, nu, nu)
        fourth_order_nu_nu_nu_nu = torch.einsum("i,j,k,l->ijkl", nu, nu, nu, nu)
        fourth_order_Pi_Pi = torch.einsum("ij,kl->ijkl", Pi_inv_1_2, Pi_inv_1_2)
        fourth_order_Pi2_Pi2 = torch.einsum("ij,kl->ijkl", Pi_inv_2_2, Pi_inv_2_2)
        
        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))
        R12_prime_mu = nu*R12
        R12_prime_epsilon = -0.5*(torch.linalg.inv(Pi_1_2) - nu @ nu.T)*R12
        R1_prime_2_prime_mu_mu = -2*R12_prime_epsilon
        R1_prime_2_prime_mu_epsilon = 0.5*R12*(third_order_Pi_nu + torch.einsum("ijk->jki" ,third_order_Pi_nu) + torch.einsum("ijk->kij" ,third_order_Pi_nu) - third_order_nu_nu_nu)
        R1_prime_2_prime_epsilon_epsilon = 0.25*R12*(fourth_order_nu_nu_nu_nu
                                                    - fourth_order_Pi_nu_nu - torch.einsum("ijkl->ikjl" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->iljk" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jkil" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jlik" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->klij" ,fourth_order_Pi_nu_nu)
                                                    + fourth_order_Pi_Pi + torch.einsum("ijkl->ikjl" ,fourth_order_Pi_Pi) + torch.einsum("ijkl->iljk" ,fourth_order_Pi_Pi))

        R22_prime_epsilon = -0.5*R22*Pi_inv_2_2
        R2_prime_2_prime_mu_mu = Pi_inv_2_2 * R22
        R2_prime_2_prime_epsilon_epsilon = (fourth_order_Pi2_Pi2 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi2_Pi2) + torch.einsum("ijkl->iljk" ,fourth_order_Pi2_Pi2)) * R22

        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.d)
        Tau2_mu = torch.linalg.solve(Epsilon2 + self.gp_covariance, diff_data2 * normal_data_unsqueezed2, left=False)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.d)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        Tau_epsilon2 = 0.5*torch.linalg.solve(Epsilon2 + self.gp_covariance, torch.linalg.solve(Epsilon2 + self.gp_covariance, outer_prod2, left=False))
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + t2X @ self.inverse_data_covar_y
        mean_prime_2 = (self.d_mu * (Tau2_mu.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (Tau_epsilon2.T @ self.inverse_data_covar_y)).sum()

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - self.t_1X @ inverse_data_covar_t2X
        
        # Compute Var 
        cov_2_2_prime = (self.d_epsilon*R22_prime_epsilon).sum() - (self.d_mu * (Tau2_mu.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_t2X)).sum()

        product_R12_prime_mu, product_R12_prime_epsilon = (self.d_mu*R12_prime_mu).sum(), (self.d_epsilon*R12_prime_epsilon).sum()
        cov_1_2_prime = product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (Tau2_mu.T @ self.inverse_data_covar_t1X)).sum() - (self.d_epsilon * (Tau_epsilon2.T @ self.inverse_data_covar_t1X)).sum()
        #R21_prime_mu = - R12_prime_mu and R21_prime_epsilon = R12_prime_epsilon
        cov_2_1_prime = -product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (self.Tau_mu1.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ inverse_data_covar_t2X)).sum()
        
        ## Compute fourth term
        cov_1_prime_2_prime = self.d_mu @ R1_prime_2_prime_mu_mu @ self.d_mu \
                            + 2*((self.d_mu @ R1_prime_2_prime_mu_epsilon) * self.d_epsilon).sum() \
                            + ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R1_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum() \
                            - self.d_mu @ self.Tau_mu1.T @ self.inverse_data @ Tau2_mu @ self.d_mu - (self.d_epsilon*(self.Tau_epsilon1.T @ self.inverse_data @ Tau2_mu @ self.d_mu)).sum() - (self.d_epsilon*(Tau_epsilon2.T @ self.inverse_data @ self.Tau_mu1 @ self.d_mu)).sum() \
                            - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", self.Tau_epsilon1.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()
        
        var_prime_2 = self.d_mu @ R2_prime_2_prime_mu_mu @ self.d_mu \
                    + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R2_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.d,self.d,1,1))).sum()

        mean_joint = torch.tensor([self.mean_1, self.mean_prime_1, mean_2,  mean_prime_2], dtype=self.train_X.dtype, device=self.train_X.device)
        covar_joint = torch.tensor([[self.var_1, self.cov_1_1_prime, cov_1_2, cov_1_2_prime],
                                    [self.cov_1_1_prime, self.var_prime_1, cov_2_1_prime, cov_1_prime_2_prime],
                                    [cov_1_2, cov_2_1_prime, var_2, cov_2_2_prime],
                                    [cov_1_2_prime, cov_1_prime_2_prime, cov_2_2_prime, var_prime_2]], dtype=self.train_X.dtype, device=self.train_X.device)
        return mean_joint.detach(), covar_joint.detach()
    
    def compute_p_wolfe(self, t):
        result = self.compute_joint_distribution(t)
        if result:
            mean_joint, covar_joint = result
            mean_joint = -mean_joint ## wolfe is for minimization, we maximize by default
            return self.compute_wolfe(mean_joint, covar_joint, t)
        else:
            return 0

    def compute_p_armijo(self, t):
        result = self.compute_joint_distribution_first_order(t)
        if result:
            mean_joint, covar_joint = result
            mean_joint = -mean_joint ## wolfe is for minimization, we maximize by default
            return self.compute_armijo(mean_joint, covar_joint, t)
        else:
            return 0.

class QuadratureExploration(AnalyticAcquisitionFunction):
    r"""Single-outcome Quadrature bRT.
    Quadrature variance minimization acquisitiion function
    """
    def __init__(
        self,
        model: Model,
        distribution: MultivariateNormal,
        batch_acq = 5,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ):
        """Acquisition function for bayesian quadrature

        Args:
            model (Model): Surrogate model
            distribution (MultivariateNormal): Quadrature distribution
            batch_acq (int, optional): Batches evaluated, different from the batch size of points evalauted by the objective. Defaults to 5.
            posterior_transform (Optional[PosteriorTransform], optional): _description_. Defaults to None.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.batch_size = batch_acq
        self.distribution=distribution
        self.train_X = self.model.train_inputs[0]
        self.gp_covariance = torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone())
        self.noise_X_X = self.model.likelihood.noise.detach().clone() * torch.eye(self.train_X.shape[0], dtype=self.train_X.dtype, device=self.train_X.device)
        self.cov_X_X = (self.model.covar_module(self.train_X) + self.noise_X_X).evaluate()
        self.cov_X_X_inv = torch.linalg.inv(self.cov_X_X)
        self.t_X_train = torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.distribution.covariance_matrix + self.gp_covariance).log_prob(self.train_X))
        self.t_X_train_batch = self.t_X_train[None, :, None].repeat(self.batch_size,1,1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Exploration Quadrature value
        """
        B = self.model.covar_module(self.train_X, X).evaluate()
        C = torch.einsum("ijk -> ikj", B)
        D = self.model.covar_module(X, X).evaluate() + (self.model.likelihood.noise.detach().clone() * torch.eye(X.shape[1], dtype=self.train_X.dtype, device=self.train_X.device))[None, :, :]

        t_X = torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.distribution.covariance_matrix + self.gp_covariance).log_prob(X))
        
        #Compute efficient inverse:
        interm = torch.linalg.inv(D - torch.bmm((C @ self.cov_X_X_inv), B))
        P_inv_A = self.cov_X_X_inv + self.cov_X_X_inv @ (B @ (interm @ (C @ self.cov_X_X_inv)))
        P_inv_B = - self.cov_X_X_inv @ B @ interm
        P_inv_C = - interm @ C @ self.cov_X_X_inv
        P_inv_D = interm

        #For debugging
        batch_size = X.shape[0]
        train_X_batch = torch.unsqueeze(self.train_X, 0).repeat(batch_size, 1, 1)
        X_full = torch.cat((train_X_batch, X), dim= 1)
        gp_kernel = torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone())
        noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(X[0].shape[0] + self.train_X.shape[0], dtype=X.dtype, device=X.device)
        t_X_full = torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.distribution.covariance_matrix + gp_kernel).log_prob(X_full))
        full_inv = torch.linalg.inv((self.model.covar_module(X_full) + noise_tensor).evaluate())
        print("Debug test")
        print("Full matrix inverse", full_inv)
        print("Formula:", P_inv_A)
        print("Formula:", P_inv_B)
        print("Formula:", P_inv_C)
        print("Formula:", P_inv_D)
        print("Check cov X")
        print((self.model.covar_module(X_full) + noise_tensor).evaluate())
        print("From formula")
        print(self.cov_X_X)

        first_term = torch.bmm(P_inv_A, self.t_X_train_batch).squeeze() @ self.t_X_train
        second_term = torch.bmm(P_inv_B, t_X[:, :, None]).squeeze() @ self.t_X_train
        third_term = (t_X[:, None, :] @ (P_inv_C @ self.t_X_train_batch)).squeeze()
        fourth_term = (t_X[:, None, :] @ (P_inv_D @ t_X[:, :, None])).squeeze()

        return first_term + second_term + third_term + fourth_term
    
class QuadratureExplorationBis(AnalyticAcquisitionFunction):
    r"""Single-outcome Quadrature bRT.
    Quadrature variance minimization acquisitiion function
    """
    def __init__(
        self,
        model: Model,
        distribution: MultivariateNormal,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ):
        r"""Single-outcome Quadrature bRT.
        Args:
            model: A fitted single-outcome model.
            distribution: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.distribution=distribution
        self.train_X = self.model.train_inputs[0]
        self.gp_covariance = (torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone()))**2
        self.constant = self.model.covar_module.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Exploration Quadrature value
        """
        batch_size = X.shape[0]
        train_X_batch = torch.unsqueeze(self.train_X, 0).repeat(batch_size, 1, 1)
        X_full = torch.cat((train_X_batch, X), dim= 1)
        noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(X[0].shape[0] + self.train_X.shape[0], dtype=X.dtype, device=X.device)
        t_X = self.constant * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.distribution.covariance_matrix + self.gp_covariance).log_prob(X_full))
        v = torch.linalg.solve((self.model.covar_module(X_full) + noise_tensor).evaluate(), t_X)
        return (v * t_X).sum(dim=-1)

    


if __name__ == "__main__":
    # from botorch import fit_gpytorch_mll
    # from gpytorch.mlls import ExactMarginalLogLikelihood

    # quad_distrib = MultivariateNormal(torch.tensor([0., 0.]), torch.diag(torch.tensor([1., 1.])))
    # train_X = torch.rand(20, 2)
    # test_X = torch.rand(2, 2, 2)
    # train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
    # model = SingleTaskGP(train_X, train_Y)
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_mll(mll)
    # quad = QuadratureExploration(model, quad_distrib)
    # acq_2 = QuadratureExplorationBis(model=model,
    #                                 distribution= quad_distrib)
    # acq_1 = QuadratureExploration(model=model,
    #                         distribution= quad_distrib,
    #                         batch_acq = test_X.shape[0])
    
    # print("Value acquisition 1", acq_1(test_X))
    # print("Value acquisition 2", acq_2(test_X))
    from botorch import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    from gpytorch.kernels import RBFKernel
    from gpytorch.kernels.scale_kernel import ScaleKernel
    from gpytorch.priors.torch_priors import GammaPrior

    def plot_GP_fit(model, distribution, train_X, targets, obj, lb=-10., up=10.):
        """ Plot the figures corresponding to the Gaussian process fit
        """
        model.eval()
        model.likelihood.eval()
        test_x = torch.linspace(lb, up, 200, device=train_X.device, dtype=train_X.dtype)
        with torch.no_grad():
            # Make predictions
            predictions = model.likelihood(model(test_x))
            lower, upper = predictions.confidence_region()
        value_ = (obj(test_x.unsqueeze(-1))).flatten()

        plt.scatter(train_X.cpu().numpy(), targets.cpu().numpy(), color='black', label='Training data')
        plt.plot(test_x.cpu().numpy(), predictions.mean.cpu().numpy(), color='blue', label='Predictive mean')
        plt.plot(test_x.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')
        plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='lightblue', alpha=0.5, label='Confidence region')
        
        x = np.linspace(distribution.loc - 3*distribution.covariance_matrix, distribution.loc + 3*distribution.covariance_matrix, 100).flatten()
        plt.plot(x, stats.norm.pdf(x, distribution.loc, distribution.covariance_matrix).flatten())
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gaussian Process Regression')
        plt.legend()
        plt.savefig("regression.png")
    
    quad_distrib = MultivariateNormal(torch.tensor([0., 0.]), torch.diag(torch.tensor([1., 1.])))
    train_X = torch.rand(2, 2)
    test_X = torch.rand(60, 2, 2)
    train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)

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
            t_max = 10,
            budget = 50,
            maximize = True)
    
    quad.gradient_direction(sample = False)
    print("mean grad:", quad.d_mu)
    print("variance grad:", quad.d_epsilon)
    acq_2 = QuadratureExplorationBis(model=model,
                                distribution= quad_distrib)
    acq_1 = QuadratureExploration(model=model,
                            distribution= quad_distrib,
                            batch_acq = 2)

    print("Value acquisition 1", acq_1(test_X))
    print("Value acquisition 2", acq_2(test_X))

    t_linspace = torch.linspace(-1., 1., 200, dtype=train_X.dtype)
    result_wolfe = []
    for t in t_linspace:
        result_wolfe.append(quad.compute_p_wolfe(t))
    wolfe_tensor = torch.tensor(result_wolfe)

    #plot_GP_fit(model, quad_distrib, train_X, train_Y, obj = lambda x :-x**2, lb=-2., up=2.)