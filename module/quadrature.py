import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from .utils import bounded_bivariate_normal_integral, nearestPD, isPD, EI, log_EI

""" 
Class handling quadrature computations
"""

class Quadrature:
    def __init__(self, 
                 model=None
                 ) -> None:
        
        ### Create attributes
        self.model=model
        self.dim = model.train_inputs[0].shape[1]
        self.device = model.train_inputs[0].device
        self.dtype = model.train_inputs[0].dtype
            
        # Extract model quantities
        # TODO optimise the use of cache
        self.train_X = self.model.train_inputs[0]
        self.noise_tensor = self.model.likelihood.noise.detach().clone() * torch.eye(self.train_X.shape[0], dtype=self.train_X.dtype, device=self.train_X.device)
        self.gp_covariance = ((torch.diag(self.model.covar_module.base_kernel.lengthscale[0].detach().clone()))**2).detach().clone()
        self.outputscale = self.model.covar_module.outputscale.detach().clone()
        self.K_X_X = (self.model.covar_module(self.train_X) + self.noise_tensor).evaluate().detach().clone()
        self.mean_constant = self.model.mean_module.constant.detach().clone()
        self.inverse_data_covar_y = torch.linalg.solve(self.K_X_X, self.model.train_targets - self.mean_constant)
        self.inverse_data = torch.linalg.inv(self.K_X_X)
        self.constant = (self.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()

    def precompute_linesearch(self, mean, covariance, grad):
        self.d_mu, self.d_epsilon = grad[0], grad[1]
        self.mu1 = mean
        self.Epsilon1 = covariance
        
        self.covariance_gp_distr = self.gp_covariance + covariance
        self.constant = (self.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()
        
        self.t_1X = self.constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = self.covariance_gp_distr).log_prob(self.train_X))
        self.R_11 = self.constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = self.covariance_gp_distr + covariance).log_prob(mean))

        self.inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, self.t_1X)

        self.mean_1 = self.mean_constant + (self.t_1X.T @ self.inverse_data_covar_y)
        self.var_1 = self.R_11 - self.t_1X.T @ self.inverse_data_covar_t1X

        ## Precompute quantities for method of lines
        Pi_1_1 = self.gp_covariance + 2*covariance
        self.Pi_inv_1_1 = torch.linalg.inv(Pi_1_1)
        self.fourth_order_Pi1_Pi1 = torch.einsum("ij,kl->ijkl", self.Pi_inv_1_1, self.Pi_inv_1_1)

        # Compute tau matrix
        self.diff_data = (self.train_X - mean) # Nxd
        self.normal_data_unsqueezed = torch.unsqueeze(self.t_1X, -1).repeat(1,self.dim)
        self.Tau_mu1 = torch.linalg.solve(self.covariance_gp_distr, self.diff_data * self.normal_data_unsqueezed, left=False)
        
        self.normal_data_unsqueezed = torch.unsqueeze(self.normal_data_unsqueezed, -1).repeat(1,1,self.dim)
        self.outer_prod = torch.bmm(torch.unsqueeze(self.diff_data, -1), torch.unsqueeze(self.diff_data, 1))
        self.outer_prod = (self.outer_prod - self.covariance_gp_distr) * self.normal_data_unsqueezed
        self.Tau_epsilon1 = 0.5 * torch.linalg.solve(self.covariance_gp_distr, torch.linalg.solve(self.covariance_gp_distr, self.outer_prod, left=False))
        
        self.R11_prime_epsilon = -0.5*self.R_11*self.Pi_inv_1_1
        self.R1_prime_1_prime_mu_mu = self.Pi_inv_1_1 * self.R_11
        self.R1_prime_1_prime_epsilon_epsilon = (self.fourth_order_Pi1_Pi1 + torch.einsum("ijkl->ikjl" ,self.fourth_order_Pi1_Pi1) + torch.einsum("ijkl->iljk" ,self.fourth_order_Pi1_Pi1)) * self.R_11

        self.mean_prime_1 = (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_y)).sum()
        self.cov_1_1_prime = (self.d_epsilon*self.R11_prime_epsilon).sum() - (self.d_mu * (self.Tau_mu1.T @ self.inverse_data_covar_t1X)).sum() - (self.d_epsilon * (self.Tau_epsilon1.T @ self.inverse_data_covar_t1X)).sum()
        # TODO where is posterior GP term here?
        self.var_prime_1 = self.d_mu @ self.R1_prime_1_prime_mu_mu @ self.d_mu \
            + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * self.R1_prime_1_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum() \
            - self.d_mu @ self.Tau_mu1.T @ self.inverse_data @ self.Tau_mu1 @ self.d_mu - 2*(self.d_epsilon*(self.Tau_epsilon1.T @ self.inverse_data @ self.Tau_mu1 @ self.d_mu)).sum() \
            - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", self.Tau_epsilon1.T, torch.einsum("ij, jkl->ikl", self.inverse_data, self.Tau_epsilon1)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum()
    
    def jointdistribution_linesearch(self, mean, covariance):
        ## Same function as joint distribution function but with precomputed terms
        if not isPD(covariance):
            covariance = nearestPD(covariance)
            return None, None
        
        mu2, Epsilon2 = mean, covariance
        Pi_1_2 = self.gp_covariance + self.Epsilon1 + Epsilon2
        Pi_2_2 = self.gp_covariance + 2*Epsilon2
        Pi_inv_1_2 = torch.linalg.inv(Pi_1_2)
        Pi_inv_2_2 = torch.linalg.inv(Pi_2_2)
        
        nu = Pi_inv_1_2 @ (self.mu1 - mu2)
        
        third_order_Pi_nu = torch.einsum("i,jk->ijk", nu, Pi_inv_1_2)
        third_order_nu_nu_nu = torch.einsum("i,j,k->ijk", nu, nu, nu)

        fourth_order_Pi_nu_nu = torch.einsum("ij,k,l->ijkl", Pi_inv_1_2, nu, nu)
        fourth_order_nu_nu_nu_nu = torch.einsum("i,j,k,l->ijkl", nu, nu, nu, nu)
        fourth_order_Pi_Pi = torch.einsum("ij,kl->ijkl", Pi_inv_1_2, Pi_inv_1_2)
        fourth_order_Pi2_Pi2 = torch.einsum("ij,kl->ijkl", Pi_inv_2_2, Pi_inv_2_2)
        
        R22 = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix = Pi_2_2).log_prob(mu2))
        R12 = self.constant * torch.exp(MultivariateNormal(loc = self.mu1, covariance_matrix = Pi_1_2).log_prob(mu2))
        R12_prime_mu = nu*R12
        R12_prime_epsilon = -0.5*(torch.linalg.inv(Pi_1_2) - nu @ nu.T)*R12
        R1_prime_2_prime_mu_mu = -2*R12_prime_epsilon
        R1_prime_2_prime_mu_epsilon = 0.5*R12*(third_order_Pi_nu + torch.einsum("ijk->jki" ,third_order_Pi_nu) + torch.einsum("ijk->kij" ,third_order_Pi_nu) - third_order_nu_nu_nu)
        R1_prime_2_prime_epsilon_epsilon = 0.25*R12*(fourth_order_nu_nu_nu_nu
                                                    - fourth_order_Pi_nu_nu - torch.einsum("ijkl->ikjl" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->iljk" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jkil" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->jlik" ,fourth_order_Pi_nu_nu) - torch.einsum("ijkl->klij" ,fourth_order_Pi_nu_nu)
                                                    + fourth_order_Pi_Pi + torch.einsum("ijkl->ikjl" ,fourth_order_Pi_Pi) + torch.einsum("ijkl->iljk" ,fourth_order_Pi_Pi))

        R22_prime_epsilon = -0.5*R22*Pi_inv_2_2
        R2_prime_2_prime_mu_mu = Pi_inv_2_2 * R22
        R2_prime_2_prime_epsilon_epsilon = 0.25*(fourth_order_Pi2_Pi2 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi2_Pi2) + torch.einsum("ijkl->iljk" ,fourth_order_Pi2_Pi2)) * R22

        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.dim)
        Tau2_mu = torch.linalg.solve(Epsilon2 + self.gp_covariance, diff_data2 * normal_data_unsqueezed2, left=False)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.dim)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        Tau_epsilon2 = 0.5*torch.linalg.solve(Epsilon2 + self.gp_covariance, torch.linalg.solve(Epsilon2 + self.gp_covariance, outer_prod2, left=False))
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + (t2X.T @ self.inverse_data_covar_y)
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
                            + ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R1_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum() \
                            - self.d_mu @ self.Tau_mu1.T @ self.inverse_data @ Tau2_mu @ self.d_mu - (self.d_epsilon*(self.Tau_epsilon1.T @ self.inverse_data @ Tau2_mu @ self.d_mu)).sum() - (self.d_epsilon*(Tau_epsilon2.T @ self.inverse_data @ self.Tau_mu1 @ self.d_mu)).sum() \
                            - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", self.Tau_epsilon1.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum()
        
        var_prime_2 = self.d_mu @ R2_prime_2_prime_mu_mu @ self.d_mu \
                    + ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R2_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum() \
                    - self.d_mu @ Tau2_mu.T @ self.inverse_data @ Tau2_mu @ self.d_mu - 2*(self.d_epsilon*(Tau_epsilon2.T @ self.inverse_data @ Tau2_mu @ self.d_mu)).sum() \
                    - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", Tau_epsilon2.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum()

        mean_joint = torch.tensor([self.mean_1, self.mean_prime_1, mean_2,  mean_prime_2], dtype=self.train_X.dtype, device=self.train_X.device)
        covar_joint = torch.tensor([[self.var_1, self.cov_1_1_prime, cov_1_2, cov_1_2_prime],
                                    [self.cov_1_1_prime, self.var_prime_1, cov_2_1_prime, cov_1_prime_2_prime],
                                    [cov_1_2, cov_2_1_prime, var_2, cov_2_2_prime],
                                    [cov_1_2_prime, cov_1_prime_2_prime, cov_2_2_prime, var_prime_2]], dtype=self.train_X.dtype, device=self.train_X.device)
        return mean_joint.detach(), covar_joint.detach()


    def quadrature(self, mean, covariance):
        ### quadrature function for optimization and gradient differentiation
        covariance_gp_distr = self.gp_covariance + covariance
        constant = (self.outputscale * torch.sqrt(torch.linalg.det(2*torch.pi*self.gp_covariance))).detach().clone()
        
        t_1X = constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = covariance_gp_distr).log_prob(self.train_X))
        R_11 = constant * torch.exp(MultivariateNormal(loc = mean, covariance_matrix = covariance_gp_distr + covariance).log_prob(mean))

        inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, t_1X) #Independant of t!..

        # mean_1 = self.train_y_standardized_std*(self.mean_constant + (t_1X.T @ self.inverse_data_covar_y)) + self.train_y_standardized_mean
        # var_1 = (self.train_y_standardized_std**2)*(R_11 - t_1X.T @ inverse_data_covar_t1X)

        mean_1 = self.mean_constant + (t_1X.T @ self.inverse_data_covar_y)
        var_1 = R_11 - t_1X.T @ inverse_data_covar_t1X
        
        return mean_1, var_1
    
    def compute_joint_distribution(self, mu1, Epsilon1, mu2, Epsilon2, grad):
        
        """Computes joint distribution between [g(0), g'(0), g(t), g'(t)], only works for cmaes"""
        self.d_mu, self.d_epsilon = grad[0], grad[1]
        
        t_1X = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = self.gp_covariance + Epsilon1).log_prob(self.train_X))
        R_11 = self.constant * torch.exp(MultivariateNormal(loc = mu1, covariance_matrix = self.gp_covariance + 2*Epsilon1).log_prob(mu1))

        inverse_data_covar_t1X = torch.linalg.solve(self.K_X_X, t_1X)

        mean_1 = self.mean_constant + (t_1X.T @ self.inverse_data_covar_y)
        var_1 = R_11 - t_1X.T @ inverse_data_covar_t1X

        ## Precompute quantities for method of lines
        Pi_1_1 = self.gp_covariance + 2*Epsilon1
        Pi_inv_1_1 = torch.linalg.inv(Pi_1_1)
        fourth_order_Pi1_Pi1 = torch.einsum("ij,kl->ijkl", Pi_inv_1_1, Pi_inv_1_1)

        # Compute tau matrix
        diff_data = (self.train_X - mu1) # Nxd
        normal_data_unsqueezed = torch.unsqueeze(t_1X, -1).repeat(1,self.dim)
        Tau_mu1 = torch.linalg.solve(self.gp_covariance + Epsilon1, diff_data * normal_data_unsqueezed, left=False)
        
        normal_data_unsqueezed = torch.unsqueeze(normal_data_unsqueezed, -1).repeat(1,1,self.dim)
        outer_prod = torch.bmm(torch.unsqueeze(diff_data, -1), torch.unsqueeze(diff_data, 1))
        outer_prod = (outer_prod - (self.gp_covariance + Epsilon1)) * normal_data_unsqueezed
        Tau_epsilon1 = 0.5 * torch.linalg.solve(self.gp_covariance + Epsilon1, torch.linalg.solve(self.gp_covariance + Epsilon1, outer_prod, left=False))
        
        R11_prime_epsilon = -0.5*R_11*Pi_inv_1_1
        R1_prime_1_prime_mu_mu = Pi_inv_1_1 * R_11
        R1_prime_1_prime_epsilon_epsilon = (fourth_order_Pi1_Pi1 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi1_Pi1) + torch.einsum("ijkl->iljk" ,fourth_order_Pi1_Pi1)) * R_11

        mean_prime_1 = (self.d_mu * (Tau_mu1.T @ self.inverse_data_covar_y)).sum() + (self.d_epsilon * (Tau_epsilon1.T @ self.inverse_data_covar_y)).sum()
        cov_1_1_prime = (self.d_epsilon*R11_prime_epsilon).sum() - (self.d_mu * (Tau_mu1.T @ inverse_data_covar_t1X)).sum() - (self.d_epsilon * (Tau_epsilon1.T @ inverse_data_covar_t1X)).sum()
        # TODO where is posterior GP term here?
        var_prime_1 = self.d_mu @ R1_prime_1_prime_mu_mu @ self.d_mu \
            + 0.25*((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * R1_prime_1_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum() \
            - self.d_mu @ Tau_mu1.T @ self.inverse_data @ Tau_mu1 @ self.d_mu - 2*(self.d_epsilon*(Tau_epsilon1.T @ self.inverse_data @ Tau_mu1 @ self.d_mu)).sum() \
            - ((torch.unsqueeze(torch.unsqueeze(self.d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", Tau_epsilon1.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon1)) * (torch.unsqueeze(torch.unsqueeze(self.d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum()

        if not isPD(Epsilon2):
            Epsilon2 = nearestPD(Epsilon2)
            return None, None
        d_mu, d_epsilon = grad[0], grad[1]
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
        R2_prime_2_prime_epsilon_epsilon = 0.25*(fourth_order_Pi2_Pi2 + torch.einsum("ijkl->ikjl" ,fourth_order_Pi2_Pi2) + torch.einsum("ijkl->iljk" ,fourth_order_Pi2_Pi2)) * R22

        t2X = self.constant * torch.exp(MultivariateNormal(loc = mu2, covariance_matrix=Epsilon2 + self.gp_covariance).log_prob(self.train_X))

        # Compute Tau mu 2
        diff_data2 = (self.train_X - mu2) # Nxd
        normal_data_unsqueezed2 = torch.unsqueeze(t2X, -1).repeat(1,self.dim)
        Tau2_mu = torch.linalg.solve(Epsilon2 + self.gp_covariance, diff_data2 * normal_data_unsqueezed2, left=False)

        # Compute Tau Epsilon2
        normal_data_unsqueezed2 = torch.unsqueeze(normal_data_unsqueezed2, -1).repeat(1,1,self.dim)
        outer_prod2 = torch.bmm(torch.unsqueeze(diff_data2, -1), torch.unsqueeze(diff_data2, 1))
        outer_prod2 = (outer_prod2 - Epsilon2 - self.gp_covariance) * normal_data_unsqueezed2
        
        Tau_epsilon2 = 0.5*torch.linalg.solve(Epsilon2 + self.gp_covariance, torch.linalg.solve(Epsilon2 + self.gp_covariance, outer_prod2, left=False))
        #Compute posterior term
        inverse_data_covar_t2X = torch.linalg.solve(self.K_X_X, t2X) 

        # Compute covariance structure
        ## Compute mean elements
        mean_2 = self.model.mean_module.constant + (t2X.T @ self.inverse_data_covar_y)
        mean_prime_2 = (d_mu * (Tau2_mu.T @ self.inverse_data_covar_y)).sum() + (d_epsilon * (Tau_epsilon2.T @ self.inverse_data_covar_y)).sum()

        ## Compute Var elements
        var_2 = R22 - t2X @ inverse_data_covar_t2X
        cov_1_2 = R12 - t_1X @ inverse_data_covar_t2X
        
        # Compute Var 
        cov_2_2_prime = (d_epsilon*R22_prime_epsilon).sum() - (d_mu * (Tau2_mu.T @ inverse_data_covar_t2X)).sum() - (d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_t2X)).sum()

        product_R12_prime_mu, product_R12_prime_epsilon = (d_mu*R12_prime_mu).sum(), (d_epsilon*R12_prime_epsilon).sum()
        cov_1_2_prime = product_R12_prime_mu + product_R12_prime_epsilon - (d_mu * (Tau2_mu.T @ inverse_data_covar_t1X)).sum() - (d_epsilon * (Tau_epsilon2.T @ inverse_data_covar_t1X)).sum()
        #R21_prime_mu = - R12_prime_mu and R21_prime_epsilon = R12_prime_epsilon
        cov_2_1_prime = -product_R12_prime_mu + product_R12_prime_epsilon - (d_mu * (Tau_mu1.T @ inverse_data_covar_t2X)).sum() - (d_epsilon * (Tau_epsilon1.T @ inverse_data_covar_t2X)).sum()
        
        ## Compute fourth term
        cov_1_prime_2_prime = d_mu @ R1_prime_2_prime_mu_mu @ d_mu \
                            + 2*((d_mu @ R1_prime_2_prime_mu_epsilon) * d_epsilon).sum() \
                            + ((torch.unsqueeze(torch.unsqueeze(d_epsilon, -1), -1)) * R1_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum() \
                            - d_mu @ Tau_mu1.T @ self.inverse_data @ Tau2_mu @ d_mu - (d_epsilon*(Tau_epsilon1.T @ self.inverse_data @ Tau2_mu @ d_mu)).sum() - (d_epsilon*(Tau_epsilon2.T @ self.inverse_data @ Tau_mu1 @ d_mu)).sum() \
                            - ((torch.unsqueeze(torch.unsqueeze(d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", Tau_epsilon1.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum()
        
        var_prime_2 = d_mu @ R2_prime_2_prime_mu_mu @ d_mu \
                    + ((torch.unsqueeze(torch.unsqueeze(d_epsilon, -1), -1)) * R2_prime_2_prime_epsilon_epsilon * (torch.unsqueeze(torch.unsqueeze(d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum() \
                    - d_mu @ Tau2_mu.T @ self.inverse_data @ Tau2_mu @ d_mu - 2*(d_epsilon*(Tau_epsilon2.T @ self.inverse_data @ Tau2_mu @ d_mu)).sum() \
                    - ((torch.unsqueeze(torch.unsqueeze(d_epsilon, -1), -1)) * torch.einsum("ijk, klm->ijlm", Tau_epsilon2.T, torch.einsum("ij, jkl->ikl", self.inverse_data, Tau_epsilon2)) * (torch.unsqueeze(torch.unsqueeze(d_epsilon, 0), 0).repeat(self.dim,self.dim,1,1))).sum()

        mean_joint = torch.tensor([mean_1, mean_prime_1, mean_2,  mean_prime_2], dtype=self.train_X.dtype, device=self.train_X.device)
        covar_joint = torch.tensor([[var_1, cov_1_1_prime, cov_1_2, cov_1_2_prime],
                                    [cov_1_1_prime, var_prime_1, cov_2_1_prime, cov_1_prime_2_prime],
                                    [cov_1_2, cov_2_1_prime, var_2, cov_2_2_prime],
                                    [cov_1_2_prime, cov_1_prime_2_prime, cov_2_2_prime, var_prime_2]], dtype=self.train_X.dtype, device=self.train_X.device)
        return mean_joint.detach(), covar_joint.detach()