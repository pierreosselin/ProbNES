## Generate objective

from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Rastrigin
from .utils import Sphere
import gpytorch
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets  # transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def get_pretrained_dir() -> str:
    return os.path.join(os.getcwd(), "data/pretrained_models")

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Objective:
    def __init__(self,
                 obj_func: Callable,
                 dim: int,
                 device: Any,
                 dtype: Any,
                 bounds: torch.Tensor,
                 noise_std: Optional[float] = None,
                 best_value: Optional[float] = None,
                 negate: bool = False):
        self.obj_func = obj_func
        self.noise_std = noise_std
        self.best_value = best_value
        self.negate = negate
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.bounds = bounds

    @torch.no_grad()
    def evaluate_true(self, X):
        if isinstance(X, torch.Tensor):
            if self.negate:
                return -self.obj_func(X)
            else:
                return self.obj_func(X)
        else:
            raise TypeError("Only torch tensor are allowed")

    @property
    def ground_truth(self):
        return self.best_value

    def raw_to_scaled(self):
        raise NotImplementedError

    def scaled_to_raw(self):
        raise NotImplementedError

    def __call__(self, X: torch.Tensor, noise: bool = False):
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        return f
    
def get_objective(
        label: str,
        device: Any,
        dtype: Any,
        problem_kwargs: Dict,
) -> Objective:
    problem_kwargs = problem_kwargs or {}
    if label == "test_function":
        test_function = problem_kwargs.get("function", "rosenbrock")
        dim = problem_kwargs.get("dim", 2)
        noise_std = problem_kwargs.get("noise_std", 0.)
        initial_bounds = problem_kwargs.get("initial_bounds", 1.)
        bounds = torch.tensor([[-initial_bounds] * dim, [initial_bounds] * dim], device=device, dtype=dtype)
        if test_function == "rosenbrock":
            obj = Objective(obj_func=Rosenbrock(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "ackley":
            obj = Objective(obj_func=Ackley(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "rastrigin":
            obj = Objective(obj_func=Rastrigin(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "sphere":
            obj = Objective(obj_func=Sphere(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "function_1":
            obj_function = lambda x: torch.sin(x - 4.) + torch.sin((10./3.)*(x - 4.))
            obj = Objective(obj_func=obj_function, dim=1, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "mountains":
            obj_function = lambda x: torch.flatten(5*torch.exp(-2*(x - 1)**2) + 5*torch.exp(-2*(x + 1)**2))
            obj = Objective(obj_func=obj_function, dim=1, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=5., negate=False)
        elif test_function == "sin3":
            obj_function = lambda x: (-(1.4 - 3*(x/15+0.6))*torch.sin(18*(x/15+0.6))).flatten()
            dim = 1
            obj = Objective(obj_func=obj_function, dim=1, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=1.6, negate=True)
        else:
            raise NotImplementedError(f"Function {test_function} is not implemented")
    
    elif label == "airfoil":
        # Devices and dtype
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.double
        
        ## Load data
        train_x, train_y = torch.load('data/airfoil_scaled_train_x.pt').to(dtype=dtype, device=device), torch.load('data/airfoil_scaled_train_y.pt').to(dtype=dtype, device=device)
        state_dict = torch.load('data/airfoil_model_state.pth')
        n = train_x.shape[0]
        noises = torch.ones(n, dtype=dtype, device=device) * 1e-5
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
        #model = ExactGPModel(train_x, train_y, likelihood=likelihood)  # Create a new GP model
        model.load_state_dict(state_dict)
        model = model.to(dtype=dtype, device=device)
        target_value = train_y.cpu().max()
        
        ## Create objective
        def objec(x):
            x = x.clone().detach()
            if x.ndim == 1:
                x = x.reshape(-1,1)
            model.eval()
            likelihood.eval()

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(x))
            return observed_pred.mean
        
        obj = Objective(obj_func=objec, noise_std=0., best_value=target_value)
        return obj

    elif label == "latent_space":
        model = problem_kwargs.get("function", "mnist")
        noise_std = problem_kwargs.get("noise_std", 0.)
        initial_bounds = problem_kwargs.get("initial_bounds", 1.)
        if model == "mnist":
            dim = 20
            bounds = torch.tensor([[-initial_bounds] * dim, [initial_bounds] * dim], device=device, dtype=dtype)

            cnn_weights_path = os.path.join(get_pretrained_dir(), "mnist_cnn.pt")
            cnn_model = Net().to(dtype=dtype, device=device)
            cnn_state_dict = torch.load(cnn_weights_path, map_location=device, weights_only=True)
            cnn_model.load_state_dict(cnn_state_dict)

            vae_weights_path = os.path.join(get_pretrained_dir(), "mnist_vae.pt")
            vae_model = VAE().to(dtype=dtype, device=device)
            vae_state_dict = torch.load(vae_weights_path, map_location=device, weights_only=True)
            vae_model.load_state_dict(vae_state_dict)

            def score(y):
                """Returns a 'score' for each digit from 0 to 9. It is modeled as a squared exponential
                centered at the digit '3'.
                """
                return torch.exp(-2 * (y - 3) ** 2)
            
            def score_image(x):
                """The input x is an image and an expected score 
                based on the CNN classifier and the scoring 
                function is returned.
                """
                with torch.no_grad():
                    probs = torch.exp(cnn_model(x))  # b x 10
                    scores = score(
                        torch.arange(10, device=device, dtype=dtype)
                    ).expand(probs.shape)
                return (probs * scores).sum(dim=1)
            
            def decode(train_x):
                if train_x.ndim == 1:
                    train_x = train_x.reshape(1,-1)
                with torch.no_grad():
                    decoded = vae_model.decode(train_x)
                return decoded.view(train_x.shape[0], 1, 28, 28)
            
            def objective(x):
                return score_image(decode(x))
            obj = Objective(obj_func=objective, dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=1., negate=False)
            return obj
        
    else:
        raise NotImplementedError(f"Problem {label} is not implemented")
    return obj