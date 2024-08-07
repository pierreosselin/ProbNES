## Generate objective

from typing import Optional, Any, Union, Tuple, Callable, Dict
import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Rastrigin
from .utils import Sphere
import gpytorch
import os

from module.model import Net, VAE, get_pretrained_dir,Discriminator, Generator, cifar10

class Objective:
    def __init__(self,
                 label: str,
                 obj_func: Callable,
                 dim: int,
                 device: Any,
                 dtype: Any,
                 bounds: torch.Tensor,
                 noise_std: Optional[float] = None,
                 best_value: Optional[float] = None,
                 negate: bool = False):
        self.label = label
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

    def __call__(self, X: torch.Tensor, noise: bool = True):
        f = self.evaluate_true(X=X)
        if noise and self.noise_std > 0.:
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
            obj = Objective(label=label, obj_func=Rosenbrock(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "ackley":
            obj = Objective(label=label, obj_func=Ackley(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "rastrigin":
            obj = Objective(label=label, obj_func=Rastrigin(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "sphere":
            obj = Objective(label=label, obj_func=Sphere(dim), dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "function_1":
            obj_function = lambda x: torch.sin(x - 4.) + torch.sin((10./3.)*(x - 4.))
            obj = Objective(label=label, obj_func=obj_function, dim=1, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=True)
        elif test_function == "mountains":
            obj_function = lambda x: torch.flatten(5*torch.exp(-2*(x - 1)**2) + 5*torch.exp(-2*(x + 1)**2))
            obj = Objective(label=label, obj_func=obj_function, dim=1, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=5., negate=False)
        elif test_function == "sin3":
            obj_function = lambda x: (-(1.4 - 3*(x/15+0.6))*torch.sin(18*(x/15+0.6))).flatten()
            dim = 1
            obj = Objective(label=label, obj_func=obj_function, dim=1, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=1.6, negate=True)
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
        
        obj = Objective(label=label, obj_func=objec, noise_std=0., best_value=target_value)
        return obj

    elif label == "latent_space":
        model = problem_kwargs.get("function", "mnist")
        noise_std = problem_kwargs.get("noise_std", 0.)
        initial_bounds = problem_kwargs.get("initial_bounds", 1.)
        if model == "mnist":
            label = problem_kwargs.get("label", 3)
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
                # return -torch.abs(y - 1)
                return torch.exp(-2 * (y - label) ** 2)
            
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
            obj = Objective(label=label, obj_func=objective, dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=False)
            obj.decode = lambda x: decode(x)
            return obj
        
        elif model == "cifar10":
            label = problem_kwargs.get("label", 3)
            dim = 100
            bounds = torch.tensor([[-initial_bounds] * dim, [initial_bounds] * dim], device=device, dtype=dtype)

            D = Discriminator(ngpu=1).eval()
            G = Generator(ngpu=1).eval()

            # load weights
            D.load_state_dict(torch.load(os.path.join(get_pretrained_dir(), "netD_epoch_199.pth")))
            G.load_state_dict(torch.load(os.path.join(get_pretrained_dir(), "netG_epoch_199.pth")))
            if torch.cuda.is_available():
                D = D.cuda()
                G = G.cuda()

            net = cifar10(128, pretrained=True).eval()  
            net = net.to(device)

            def score(y):
                """Returns a 'score' for each digit from 0 to 9. It is modeled as a squared exponential
                centered at the digit '3'.
                """
                # return -torch.abs(y - 1)
                return torch.exp(-2 * (y - label) ** 2)
            
            def score_image(x):
                """The input x is an image and an expected score 
                based on the CNN classifier and the scoring 
                function is returned.
                """
                with torch.no_grad():
                    probs = net(x).softmax(dim=1)  # b x 10
                    scores = score(
                        torch.arange(10, device=device, dtype=dtype)
                    ).expand(probs.shape)
                return (probs * scores).sum(dim=1)
            
            def decode(train_x):
                if train_x.ndim == 1:
                    train_x = train_x.reshape(1,-1)
                train_x = train_x.unsqueeze(-1).unsqueeze(-1)
                with torch.no_grad():
                    decoded = G(train_x)
                return decoded
            
            def objective(x):
                return score_image(decode(x))
            obj = Objective(label=label, obj_func=objective, dim=dim, device=device, dtype=dtype, bounds=bounds, noise_std=noise_std, best_value=0., negate=False)
            obj.decode = lambda x: decode(x)
            return obj


    else:
        raise NotImplementedError(f"Problem {label} is not implemented")
    return obj