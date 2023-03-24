import numpy as np
from sklearn.preprocessing import StandardScaler
import gpytorch
import torch

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

## TODO Put ExactGPModel elsewhere

if __name__ == "__main__":

    # Load data
    x = np.loadtxt('data/airfoil_self_noise.dat')
    scale= StandardScaler()
    scaled_data = scale.fit_transform(x)
    
    # Tensor transformation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    train_x = scaled_data[:, :5]
    train_x = torch.tensor(train_x, dtype=dtype, device=device)
    train_y = scaled_data[:, 5]
    train_y = torch.tensor(train_y, dtype=dtype, device=device)
    
    # Saving scaled data
    torch.save(train_x, './data/airfoil_scaled_train_x.pt')
    torch.save(train_y, './data/airfoil_scaled_train_y.pt')

    # Training GP surrogate on the targets
    n = train_x.shape[0]
    noises = torch.ones(n, dtype=dtype, device=device) * 1e-5
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
    model = ExactGPModel(train_x, train_y, likelihood).to(dtype=dtype, device=device)

    # Find optimal model hyperparameters
    training_iter = 500
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item()
            #model.likelihood.noise.item()
        ))
        optimizer.step()
    torch.save(model.state_dict(), 'data/airfoil_model_state.pth')