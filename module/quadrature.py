from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform

from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from typing import Callable, Optional, Tuple
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from .utils import bounded_bivariate_normal_integral

class Quadrature:
    def __init__(self, 
                 model=None,
                 distribution=None,
                 c1 = 0.1,
                 c2 = 0.2,
                 batch_size=None,
                 device=None,
                 dtype=None) -> None:
        self.model=model
        self.distribution=distribution
        self.batch_size=batch_size
        self.device=device
        self.dtype=dtype
        self.d = None
        self.c1 = c1
        self.c2 = c2
        assert 0 <= self.c1
        assert self.c1 < self.c2
        assert self.c2 <= 1

        ## Compute what is independant of t:
        self.gp_kernel = torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone())
        self.covariance_matrix = self.gp_kernel + self.distribution.covariance_matrix


    def gradient_direction(self, sample = True):
        # Compute tau matrix
        train_X = self.model.train_inputs[0]
        gp_kernel = torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone())
        covariance_matrix = gp_kernel + self.distribution.covariance_matrix

        t_X = torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = covariance_matrix).log_prob(train_X))
        
        diff_data = (train_X - self.distribution.loc) # Nxd
        normal_data_unsqueezed = torch.unsqueeze(t_X, -1).repeat(1,2)
        Tau_mu = self.model.covar_module.outputscale * torch.linalg.solve(covariance_matrix, diff_data * normal_data_unsqueezed, left=False)
        
        normal_data_unsqueezed = torch.unsqueeze(normal_data_unsqueezed, -1).repeat(1,1,2)
        outer_prod = torch.bmm(torch.unsqueeze(diff_data, -1), torch.unsqueeze(diff_data, 1))

        outer_prod = (covariance_matrix - outer_prod) * normal_data_unsqueezed
        Tau_epsilon = 0.5*self.model.covar_module.outputscale * torch.linalg.solve(covariance_matrix, torch.linalg.solve(covariance_matrix, outer_prod, left=False))
        
        #Compute the gradient of m:
        noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(train_X.shape[0], dtype=train_X.dtype, device=train_X.device)
        K_X_X = self.model.covar_module(train_X) + noise_tensor
        
        
        inverse_data_covar_y = torch.linalg.solve(K_X_X, self.model.train_targets)
        gradient_m_mu =  Tau_mu.T @ inverse_data_covar_y
        gradient_m_epsilon =  Tau_epsilon.T @ inverse_data_covar_y

        ### Sample a gradient or take the expected gradient direction
        if sample:
            inverse_data_covar_t = torch.linalg.solve(K_X_X, t_X)
            R_prime = -0.5*torch.linalg.inv(covariance_matrix + self.distribution.covariance_matrix) * torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = covariance_matrix).log_prob(self.distribution.loc))
            gradient_v_mu = Tau_mu.T @ inverse_data_covar_t
            gradient_v_epsilon = R_prime - Tau_epsilon.T @ inverse_data_covar_t
            s = torch.normal(0, 1, size=(1,), device = train_X.device, dtype=train_X.dtype)
            self.d_mu = gradient_m_mu + s*gradient_v_mu
            self.d_epsilon = gradient_m_epsilon + s*gradient_v_epsilon
        else:
            self.d_mu = gradient_m_mu
            self.d_epsilon = gradient_m_epsilon

    def compute_p_wolfe(self, t):
        # Already changed dCov and Covd here
        """Computes the probability that step size ``t`` satisfies the adjusted
        Wolfe conditions under the current GP model."""
        # Compute mu and PI
        ## TODO make sure the lengthscale parameter is at the right place and check what is independant of t to optimize computation + limit number of points in gp
        mu1, mu2 = self.distribution.loc, self.distribution.loc + t*self.d_mu
        Epsilon1, Epsilon2 = self.distribution.covariance_matrix, self.distribution.covariance_matrix + t*self.d_epsilon
        gp_kernel = torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone())
        train_X = self.model.train_inputs[0]
        Pi_1_2 = gp_kernel + Epsilon1 + Epsilon2
        Pi_1_1 = gp_kernel + 2*Epsilon1
        Pi_2_2 = gp_kernel + 2*Epsilon2
        Pi_inv_1_2 = torch.linalg.inv(Pi_1_2)
        Pi_inv_1_1 = torch.linalg.inv(Pi_1_1)
        Pi_inv_2_2 = torch.linalg.inv(Pi_2_2)
        theta = self.model.covar_module.outputscale
        
        nu = -t*Pi_inv_1_2 @ self.d_mu # (mu1 - mu2)
        
        third_order_Pi_nu = torch.einsum("i,jk->ijk", nu, Pi_inv_1_2)
        third_order_nu_nu_nu = torch.einsum("i,j,k->ijk", nu, nu, nu)

        fourth_order_Pi_nu_nu = torch.einsum("ij,k,l->ijkl", Pi_inv_1_2, nu, nu)
        fourth_order_nu_nu_nu_nu = torch.einsum("i,j,k,l->ijkl", nu, nu, nu, nu)
        fourth_order_Pi_Pi = torch.einsum("ij,kl->ijkl", Pi_inv_1_2, Pi_inv_1_2)
        fourth_order_Pi1_Pi1 = torch.einsum("ij,kl->ijkl", Pi_inv_1_1, Pi_inv_1_1)
        fourth_order_Pi2_Pi2 = torch.einsum("ij,kl->ijkl", Pi_inv_2_2, Pi_inv_2_2)
        
        R11 = torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_1).log_prob(mu1)) # Independant of t!
        R22 = torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_2).log_prob(mu2))
        R12_prime_mu = nu*R12
        R12_prime_epsilon = -0.5*(torch.linalg.inv(Pi_1_2) - nu @ nu.T)*R12
        R1_prime_2_prime_mu_mu = -2*R12_prime_epsilon
        R1_prime_2_prime_mu_epsilon = 0.5*R12*(third_order_Pi_nu + torch.einsum("ijk->jki" ,third_order_Pi_nu) + torch.einsum("ijk->kij" ,third_order_Pi_nu) - third_order_nu_nu_nu)
        R1_prime_2_prime_epsilon_epsilon = 0.25*R12*(fourth_order_nu_nu_nu_nu
                                                    - fourth_order_Pi_nu_nu - torch.einsum("ijkl->ikjl" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->iljk" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jkil" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jlik" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->klij" ,fourth_order_Pi_nu_nu)
                                                    + fourth_order_Pi_Pi + torch.einsum("ijkl->ikjl" ,fourth_order_Pi_Pi) + torch.einsum("ijkl->iljk" ,fourth_order_Pi_Pi))

        t1X = torch.exp(MultivariateNormal(loc = mu1, covariance_matrix=Epsilon1 + gp_kernel).log_prob(train_X))
        t2X = torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + gp_kernel).log_prob(train_X))

        # Compute Tau mu
        diff_data1 = (train_X - mu1) # Nxd
        normal_data_unsqueezed1 = torch.unsqueeze(t1X, -1).repeat(1,2)
        Tau1_mu = torch.linalg.solve(Epsilon1 + gp_kernel, diff_data1 * normal_data_unsqueezed1, left=False)

        diff_data2 = (train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,2)
        Tau2_mu = torch.linalg.solve(Epsilon2 + gp_kernel, diff_data2 * normal_data_unsqueezed2, left=False)

        # Compute Tau Epsilon1
        normal_data_unsqueezed1 = torch.unsqueeze(normal_data_unsqueezed1, -1).repeat(1,1,2)
        outer_prod1 = torch.bmm(torch.unsqueeze(diff_data1, -1), torch.unsqueeze(diff_data1, 1))
        outer_prod1 = (Epsilon1 + gp_kernel - outer_prod1) * normal_data_unsqueezed1
        Tau_epsilon1 = 0.5*torch.linalg.solve(Epsilon1 + gp_kernel, torch.linalg.solve(Epsilon1 + gp_kernel, outer_prod1, left=False))

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,2)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (Epsilon2 + gp_kernel - outer_prod2) * normal_data_unsqueezed2
        Tau_epsilon2 = 0.5*torch.linalg.solve(Epsilon2 + gp_kernel, torch.linalg.solve(Epsilon2 + gp_kernel, outer_prod2, left=False))

        #Compute posterior term
        noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(train_X.shape[0], dtype=train_X.dtype, device=train_X.device)
        K_X_X = (self.model.covar_module(train_X) + noise_tensor).evaluate()
        inverse_data = torch.linalg.inv(K_X_X)
        inverse_data_covar_t1X = torch.linalg.solve(K_X_X, t1X) #Independant of t!
        inverse_data_covar_t2X = torch.linalg.solve(K_X_X, t2X) 
        inverse_data_covar_y = torch.linalg.solve(K_X_X, self.model.train_targets)


        # Compute covariance structure
        ## Compute mean elements
        mean_1 = t1X @ inverse_data_covar_y
        mean_2 = t2X @ inverse_data_covar_y
        mean_prime_1 = (self.d_mu * (Tau1_mu.T @ inverse_data_covar_y)).sum() + (self.d_epsilon * (Tau_epsilon1.T @ inverse_data_covar_y)).sum()
        mean_prime_2 = (self.d_mu * (Tau2_mu.T @ inverse_data_covar_y)).sum() + (self.d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_y)).sum()

        ## Compute Var elements
        var_1 = R11 - t1X @ inverse_data_covar_t1X
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - t1X @ inverse_data_covar_t2X
        
        # Compute Var 
        cov_1_1_prime = -0.5*R11*(self.d_epsilon*Pi_inv_1_2).sum() - (self.d_mu * (Tau1_mu.T @ inverse_data_covar_t1X)).sum() - (self.d_epsilon * (Tau_epsilon1.T @ inverse_data_covar_t1X)).sum()
        cov_2_2_prime = -0.5*R22*(self.d_epsilon*Pi_inv_1_2).sum() - (self.d_mu * (Tau2_mu.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_t2X)).sum()

        product_R12_prime_mu, product_R12_prime_epsilon = (self.d_mu*R12_prime_mu).sum(), (self.d_epsilon*R12_prime_epsilon).sum()
        cov_1_2_prime = product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (Tau2_mu.T @ inverse_data_covar_t1X)).sum() - (self.d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_t1X)).sum()
        #R21_prime_mu = - R12_prime_mu and R21_prime_epsilon = R12_prime_epsilon
        cov_2_1_prime = -product_R12_prime_mu + product_R12_prime_epsilon - (self.d_mu * (Tau1_mu.T @ inverse_data_covar_t2X)).sum() - (self.d_epsilon * (Tau_epsilon1.T @ inverse_data_covar_t2X)).sum()
        
        ## Compute fourth term
        cov_1_prime_2_prime = self.d_mu @ R1_prime_2_prime_mu_mu @ self.d_mu \
                            + 2*((self.d_mu @ R1_prime_2_prime_mu_epsilon) * self.d_epsilon).sum() \
                            + ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R1_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(2,2,1,1))).sum() \
                            - self.d_mu @ Tau1_mu.T @ inverse_data @ Tau2_mu @ self.d_mu - (self.d_epsilon*(Tau_epsilon1.T @ inverse_data @ Tau2_mu @ self.d_mu)).sum() - (self.d_epsilon*(Tau_epsilon2.T @ inverse_data @ Tau1_mu @ self.d_mu)).sum() \
                            - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, lmn->ijmn", Tau_epsilon1.T, torch.einsum("ij, klm->ilm", inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(2,2,1,1))).sum()
        var_prime_1 = self.d_mu @ torch.linalg.inv(Pi_1_1) @ self.d_mu * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = Pi_1_1).log_prob(mu1)) \
                    + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * (fourth_order_Pi1_Pi1 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi1_Pi1) + torch.einsum("ijkl->iljk" ,fourth_order_Pi1_Pi1)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(2,2,1,1))).sum()

        var_prime_2 = self.d_mu @ torch.linalg.inv(Pi_2_2) @ self.d_mu * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu1)) \
                    + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * (fourth_order_Pi2_Pi2 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi2_Pi2) + torch.einsum("ijkl->iljk" ,fourth_order_Pi2_Pi2)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(2,2,1,1))).sum()

        # Compute mean and covariance matrix of the two Wolfe quantities a and b
        # Compute mean and covariance structure for f(0), f'(0), f(t), f'(t)
        A_transform = torch.tensor([[1, self.c1*t, -1, 0], [1, self.c1*t, -1, 0]], dtype=train_X.dtype, device=train_X.device)
        mean_joint = theta*torch.tensor([mean_1, mean_prime_1, mean_2,  mean_prime_2], dtype=train_X.dtype, device=train_X.device)
        covar_joint = theta*torch.tensor([[var_1, cov_1_1_prime, cov_1_2, cov_1_2_prime],
                                    [cov_1_1_prime, var_prime_1, cov_2_1_prime, cov_1_prime_2_prime],
                                    [cov_1_2, cov_2_1_prime, var_2, cov_2_2_prime],
                                    [cov_1_2_prime, cov_1_prime_2_prime, cov_2_2_prime, var_prime_2]], dtype=train_X.dtype, device=train_X.device)

        mean_wolfe = A_transform @ mean_joint
        covar_wolfe = A_transform @ covar_joint @ A_transform.T
        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        
        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        assert covar_wolfe[0,1] == covar_wolfe[1,0]
        rho = (covar_wolfe[0,1]/torch.sqrt(covar_wolfe[0,0]*covar_wolfe[1,1])).cpu()
        al = -(mean_wolfe[0]/torch.sqrt(covar_wolfe[0,0])).cpu()
        bl = -(mean_wolfe[1]/torch.sqrt(covar_wolfe[1,1])).cpu()

        return bounded_bivariate_normal_integral(rho, al, torch.inf, bl, torch.inf)




## TODO Check if can be more efficient here
class QuadratureExploration(AnalyticAcquisitionFunction):
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
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Exploration Quadrature value
        """
        
        batch_size = X.shape[0]
        train_X = self.model.train_inputs[0]
        train_X_batch = torch.unsqueeze(train_X, 0).repeat(batch_size, 1, 1)
        X_full = torch.cat((train_X_batch, X), dim= 1)
        gp_kernel = torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone())
        noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(X[0].shape[0] + train_X.shape[0], dtype=X.dtype, device=X.device)
        t_X = torch.exp(MultivariateNormal(loc = self.distribution.loc, covariance_matrix = self.distribution.covariance_matrix + gp_kernel).log_prob(X_full))
        v = torch.linalg.solve((self.model.covar_module(X_full) + noise_tensor).evaluate(), t_X)
        return (v * t_X).sum(dim=-1)
    

if __name__ == "__main__":
    train_X = torch.rand(20, 2, dtype = torch.float64)
    train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
    model = SingleTaskGP(train_X, train_Y)
    ditribution = MultivariateNormal(torch.zeros(2), torch.eye(2))
    quad = QuadratureExploration(model, ditribution)
    X = torch.rand(20, 2, dtype = torch.float64)
    quad(X)