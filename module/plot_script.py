import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
import imageio.v2 as imageio
from tqdm import tqdm
import scipy.stats as stats
from module.objective import get_objective
from .utils import standardize_return
import numpy as np
from module.quadrature import Quadrature
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from module.utils import nearestPD, EI, log_EI, EI_bivariate
from botorch.utils.transforms import standardize, normalize, unnormalize
import geoopt


"""
Scripts to produce plots from the files contained in path
Refactor in the following manner:
- have a function for:
    - plotting gp fit
    - plotting distribution (1D and 2D ellipse)
    - plot data

""" 

def plot_gp_fit(ax, model, train_X, targets, obj, batch, normalize_flag=False):
    """Plot the gp fit, normalize parameter in case of input normalization"""
    bounds = obj.bounds
    lb, up = float(bounds[0][0]), float(bounds[1][0])
    
    model.eval()
    model.likelihood.eval()

    test_x_unormalized = torch.linspace(lb, up, 200, device=train_X.device, dtype=train_X.dtype)
    if normalize_flag:
        test_x = normalize(test_x_unormalized, bounds=bounds)
    else:
        test_x = test_x_unormalized
    
    with torch.no_grad():
        # Make predictions
        predictions = model.likelihood(model(test_x))
        lower, upper = predictions.confidence_region()
    
    _, mean_Y, std_Y = standardize_return(targets)
    lower, upper = lower*float(std_Y) + float(mean_Y), upper*float(std_Y) + float(mean_Y)
    value_ = (obj(test_x_unormalized.unsqueeze(-1))).flatten()

    ax.scatter(train_X.cpu().numpy(), targets.cpu().numpy(), color='black', label='Training data')
    ax.scatter(train_X.cpu().numpy()[(-batch):], targets.cpu().numpy()[(-batch):], color='red', label='Last selected points')
    ax.plot(test_x_unormalized.cpu().numpy(), predictions.mean.cpu().numpy()*float(std_Y) + float(mean_Y), color='blue', label='Predictive mean')
    ax.plot(test_x_unormalized.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')
    ax.fill_between(test_x_unormalized.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='lightblue', alpha=0.5, label='Confidence region')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gaussian Process Regression')
    ax.legend()

def plot_distribution_1D(ax, distribution):
    x = np.linspace((distribution.loc - 3*distribution.covariance_matrix).cpu().detach().numpy(), (distribution.loc + 3*distribution.covariance_matrix).cpu().detach().numpy(), 100).flatten()
    y_lim = ax.get_ylim()
    ax.plot(x, (y_lim[1] - y_lim[0])*stats.norm.pdf(x, distribution.loc.cpu().detach().numpy(), distribution.covariance_matrix.cpu().detach().numpy()).flatten(), "k")
    
def plot_synthesis_quad(optimizer, iteration, save_path=".", standardize=True):
    iteration = optimizer.iteration
    save_path = optimizer.plot_path
    save_path_gp = os.path.join(save_path, f"fitgp/synthesis_{iteration}.png")
    bounds = optimizer.objective.bounds

    b = np.arange(float(bounds[0][0]), float(bounds[1][0]), 0.2)
    d = np.arange(0, 10, 0.5)[1:]
    B, D = np.meshgrid(b, d)
    n, m = b.shape[0], d.shape[0]
    res = torch.stack((torch.tensor(B.flatten()), torch.tensor(D.flatten())), axis = 1).cpu().numpy()
    result, result_ei, result_bivariate_ei = [], [], []
    
    for el in tqdm(res):
        post = optimizer._quadrature(torch.tensor([el[0]]).to(device=optimizer.objective.device, dtype=optimizer.objective.dtype), torch.tensor([[el[1]]]).to(device=optimizer.objective.device, dtype=optimizer.objective.dtype))
        result.append(post)
        mean_joint, covar_joint = optimizer.compute_joint_distribution_zero_order(torch.tensor([el[0]]).to(device=optimizer.objective.device, dtype=optimizer.objective.dtype), torch.tensor([[el[1]]]).to(device=optimizer.objective.device, dtype=optimizer.objective.dtype))
        result_ei.append(log_EI(mean_joint[1], covar_joint[1,1], optimizer.objective.best_value))
        result_bivariate_ei.append(EI_bivariate(mean_joint, covar_joint))

    mean = torch.tensor(result).cpu().numpy()[:,0].reshape(m,n)
    std = torch.sqrt(torch.tensor(result)).cpu().numpy()[:,1].reshape(m,n)
    ei = torch.tensor(result_ei).cpu().numpy().reshape(m,n)
    ei_bivariate = torch.tensor(result_bivariate_ei).cpu().numpy().reshape(m,n)
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))

    if optimizer.policy in ["wolfe", "armijo"]:
        t_linspace = torch.linspace(0., optimizer.t_max, optimizer.budget + 1, dtype=optimizer.train_x.dtype)[1:]
        result_wolfe, result_armijo = [], []
        for t in t_linspace:
            result_wolfe.append(optimizer.wolfe_criterion(t))
            result_armijo.append(optimizer.armijo_criterion(t))
        wolfe_tensor = torch.tensor(result_wolfe).cpu().numpy()
        armijo_tensor = torch.tensor(result_armijo).cpu().numpy()

        axs[1,0].plot(t_linspace.cpu().numpy(), wolfe_tensor, color='blue', label = "Probability Wolfe condition")
        axs[1,0].plot(t_linspace.cpu().numpy(), armijo_tensor, color='red', label = "Probability Armijo condition")
        axs[1,0].set_title("Probabilistic conditions line search")
        axs[1,0].legend()

    ## Compute gradients at multiple places
    b_grad = np.arange(float(bounds[0][0]), float(bounds[1][0]), 1)
    d_grad = np.arange(0, 3, 0.5)[1:]**2
    B_grad, D_grad = np.meshgrid(b_grad, d_grad)
    #n_grad, m_grad = b_grad.shape[0], d_grad.shape[0]
    res_grad = torch.stack((torch.tensor(B_grad.flatten()), torch.tensor(D_grad.flatten())), axis = 1).cpu().numpy()
    result_grad, max_length = [], 0
    for el in tqdm(res_grad):
        mean_distrib_grad, var_distrib_grad = torch.tensor([el[0]], dtype=optimizer.objective.dtype, device=optimizer.objective.device), torch.diag(torch.tensor([el[1]], dtype=optimizer.objective.dtype, device=optimizer.objective.device))
        manifold_point_grad = geoopt.ManifoldTensor(torch.cat((mean_distrib_grad, var_distrib_grad.flatten())), manifold=optimizer.manifold)
        manifold_point_grad.requires_grad = True
        
        m, _ = optimizer._quadrature(optimizer.manifold.take_submanifold_value(manifold_point_grad, 0), optimizer.manifold.take_submanifold_value(manifold_point_grad, 1))
        m.backward()
        
        d_manifold = manifold_point_grad.grad
        d_mu, d_epsilon = optimizer.manifold.take_submanifold_value(d_manifold, 0), optimizer.manifold.take_submanifold_value(d_manifold, 1)

        mu_grad, epsilon_grad = float(d_mu.detach().clone()), float(d_epsilon.detach().clone())
        max_length = max(max_length, np.sqrt(mu_grad**2 + epsilon_grad**2))
        result_grad.append([mu_grad, epsilon_grad])
        
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))
    lb, up = float(bounds[0][0]), float(bounds[1][0])
    axs[0,0].set_xlim(lb, up)
    plot_gp_fit(axs[0,0], optimizer.model, optimizer.train_x, optimizer.train_y, optimizer.objective, optimizer.batch_size, normalize_flag=False)
    plot_distribution_1D(axs[0,0], optimizer.distribution)

    contour1 = axs[0,1].contourf(B, D, mean)
    # Gradient
    arrow_factor = 5
    for i, el in tqdm(enumerate(res_grad)):
        axs[0, 1].arrow(el[0], el[1], arrow_factor*result_grad[i][0]/max_length, arrow_factor*result_grad[i][1]/max_length, width = 0.1)
    axs[0,1].set_xlabel(r'$\mu$')
    axs[0,1].set_ylabel(r'$\sigma^{2}$')
    axs[0,1].set_title(r"Predictive mean of $g(\theta)$")

    contour3 = axs[1,1].contourf(B, D, ei)
    axs[1,1].set_xlabel('$\mu$')
    axs[1,1].set_ylabel('$\sigma^{2}$')
    axs[1,1].set_title("Log Expected improvement")

    contour4 = axs[0,2].contourf(B, D, std)
    axs[0,2].set_xlabel('$\mu$')
    axs[0,2].set_ylabel('$\sigma^{2}$')
    axs[0,2].set_title("Predictive Std of $g(\theta)$")

    contour5 = axs[1,2].contourf(B, D, ei_bivariate)
    axs[1,2].set_xlabel('$\mu$')
    axs[1,2].set_ylabel('$\sigma^{2}$')
    axs[1,2].set_title("Bivariate Expected improvement")

    axs[1,1].scatter([float(optimizer.distribution.loc)], [float(optimizer.distribution.covariance_matrix)])
    
    ### Plot update rather than assuming wolfe
    mu2, Epsilon2 = optimizer.list_mu[-2], optimizer.list_covar[-2]
    axs[1,1].scatter([float(mu2)], [float(Epsilon2)])
    axs[1,1].arrow(float(mu2), float(Epsilon2), float(optimizer.distribution.loc) - float(mu2), float(optimizer.distribution.covariance_matrix) - float(Epsilon2), width = 0.1)
    
    fig.colorbar(contour1, ax=axs[0,1])
    fig.colorbar(contour4, ax=axs[0,2])
    fig.colorbar(contour3, ax=axs[1,1])
    fig.colorbar(contour5, ax=axs[1,2])
    fig.savefig(save_path_gp)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def ci(y, N_TRIALS):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

def plot_figure_algo(alg_dir, ax):
    data_path_seeds = [f for f in os.listdir(alg_dir) if ".pt" in f]
    data_over_seeds = []
    for _, df in enumerate(data_path_seeds):
        data_path = os.path.join(alg_dir, df)
        data = torch.load(data_path, map_location="cpu")
        data_over_seeds.append(data["regret"])
    N_TRIALS = len(data_over_seeds)
    N_BATCH = data["N_BATCH"]
    BATCH_SIZE = data["BATCH_SIZE"]
    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    label = data["label"]
    data_over_seeds = [t.detach().cpu().numpy() for t in data_over_seeds]
    y = np.asarray(data_over_seeds)
    ax.plot(iters, y.mean(axis=0), ".-", label=label )
    yerr=ci(y, N_TRIALS)
    ax.fill_between(iters, y.mean(axis=0)-yerr, y.mean(axis=0)+yerr, alpha=0.1)

def plot_figure(save_path, log_transform=False):
    exp_name = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
    for experiment in exp_name:
        exp_dir = os.path.join(save_path, experiment)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        alg_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]
        for algo in alg_name:
            alg_dir = os.path.join(exp_dir, algo)
            alg_dir_param_name = [name for name in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, name))]
            for algo_param in alg_dir_param_name:
                algo_param_dir = os.path.join(alg_dir, algo_param)
                data_path_seeds = [f for f in os.listdir(algo_param_dir) if ".pt" in f]
                data_over_seeds = []
                for _, df in enumerate(data_path_seeds):
                    data_path = os.path.join(algo_param_dir, df)
                    data = torch.load(data_path, map_location="cpu")
                    data_over_seeds.append(data["regret"])
                N_TRIALS = len(data_over_seeds)
                N_BATCH = data["N_BATCH"]
                BATCH_SIZE = data["BATCH_SIZE"]
                iters = np.arange(N_BATCH + 1) * BATCH_SIZE
                label = data["label"]
                data_over_seeds = [t.detach().cpu().numpy() for t in data_over_seeds]
                y = np.asarray(data_over_seeds)
                if log_transform:
                    ax.plot(iters, np.log(y.mean(axis=0)), ".-", label=label)
                else:
                    ax.plot(iters, y.mean(axis=0), ".-", label=label)
                yerr=ci(y, N_TRIALS)
                if log_transform:
                    ax.fill_between(iters, np.log(np.clip(y.mean(axis=0)-yerr, a_min=1e-5, a_max=None)), np.log(np.clip(y.mean(axis=0)+yerr, a_min=1e-5, a_max=None)), alpha=0.1)
                else:
                    ax.fill_between(iters, y.mean(axis=0)-yerr, y.mean(axis=0)+yerr, alpha=0.1)
        if not log_transform:    
            ax.plot([0, N_BATCH * BATCH_SIZE], [0.] * 2, 'k', label="true best objective", linewidth=2)
            ax.set_ylim(0,10.)
        ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
        #ax.set_ylim(0,10.)
        ax.legend(loc="lower right")
        if not log_transform:
            fig.savefig(os.path.join(exp_dir, f"plot_regret.pdf"))
            fig.savefig(os.path.join(exp_dir, f"plot_regret.png"))
        else:
            fig.savefig(os.path.join(exp_dir, f"plot_regret_log.pdf"))
            fig.savefig(os.path.join(exp_dir, f"plot_regret_log.png"))

def distribution_gif_2D(algo_path, objective, seed, data, ax):
    X = data["X"]
    label = data["label"]
    bounds = data["bounds"]
    BATCH_SIZE = data["BATCH_SIZE"]
    N_BATCH = data["N_BATCH"]
    if label in ["SNES", "quad"]:
        mu = data["mu"]
        sigma = data["sigma"]
    b = np.arange(-bounds, bounds, 0.05)
    d = np.arange(-bounds, bounds, 0.05)
    B, D = np.meshgrid(b, d)
    n = b.shape[0]
    res = torch.stack((torch.tensor(B.flatten()), torch.tensor(D.flatten())), axis = 1)
    nu = objective(res).numpy().reshape(n, n)

    plot_path = os.path.join(algo_path, f"{seed}")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
        
    for i in range(N_BATCH):
        if ((i+1) % 5) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.contourf(B, D, nu, levels=500)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plot_X = X[(i-1)*BATCH_SIZE:i*BATCH_SIZE].numpy()
            ax.scatter(plot_X[:,0], plot_X[:,1], s=16)
            if label in ["SNES", "quad"]:
                plot_cov_ellipse(np.diag(sigma[i].numpy()), mu[i].numpy(), nstd=1, ax=ax, facecolor="none", edgecolor = 'firebrick')
                plot_cov_ellipse(np.diag(sigma[i].numpy()), mu[i].numpy(), nstd=2, ax=ax, facecolor="none", edgecolor = 'fuchsia', linestyle='--')
                plot_cov_ellipse(np.diag(sigma[i].numpy()), mu[i].numpy(), nstd=3, ax=ax, facecolor="none", edgecolor = 'blue', linestyle=':')
            fig.savefig(os.path.join(plot_path, f"plot{i}.png"))
            plt.close()
    images = []
    for i in range(N_BATCH):
        if ((i+1) % 5) == 0:
            images.append(imageio.imread(os.path.join(plot_path, f"plot{i}.png")))
    imageio.mimsave(os.path.join(algo_path, f"gif{seed}.gif"), images)

def distribution_gif_1D(algo_path, objective, seed, data, ax):
    X = data["X"]
    label = data["label"]
    bounds = data["bounds"]
    BATCH_SIZE = data["BATCH_SIZE"]
    N_BATCH = data["N_BATCH"]
    if label in ["SNES", "quad"]:
        mu = data["mu"]
        sigma = data["sigma"]
    b = np.arange(-bounds, bounds, 0.05)
    nu = -objective(torch.tensor(b).reshape(-1,1)).cpu().numpy()
    plot_path = os.path.join(algo_path, f"{seed}")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
        
    for i in range(N_BATCH):
        if ((i+1) % 5) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(b, nu)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            plot_X = X[(i-1)*BATCH_SIZE:i*BATCH_SIZE].numpy()
            ax.scatter(plot_X[:,0], np.zeros_like(plot_X[:,0]), s=16)
            if label in ["SNES", "quad"]:
                x = np.linspace(mu[i] - 3*torch.sqrt(sigma[i]), mu[i] + 3*torch.sqrt(sigma[i]), 100).flatten()
                plt.plot(x, 0.5*(y_lim[1] - y_lim[0]) * stats.norm.pdf(x, mu[i], torch.sqrt(sigma[i])).flatten(), label = "N(" + "{:.1E}".format(float(mu[i])) + ", " + "{:.1E}".format(float(sigma[i]))+")"  )
            plt.legend()
            fig.savefig(os.path.join(plot_path, f"plot{i}.png"))
            plt.close()

    images = []
    for i in range(N_BATCH):
        if ((i+1) % 5) == 0:
            images.append(imageio.imread(os.path.join(plot_path, f"plot{i}.png")))
    imageio.mimsave(os.path.join(algo_path, f"gif{seed}.gif"), images)

def plot_distribution_gif(config, n_seeds=1):
    """
    n_seeds: Number of seeds one wants to plot the trajectory
    """
    ### Here make a loop to plot gif to all relevant configurations
    save_path = config["save_dir"]
    exp_name = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
    for experiment in exp_name:
        exp_dir = os.path.join(save_path, experiment)
        alg_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]        
        dim = int(experiment.split("_")[2][-1])
        obj = get_objective(label=config["problem_name"], device=None, dtype=None, problem_kwargs=config["problem_settings"])
        for algo in tqdm(alg_name, desc="Processing Algorithms..."):
            algo_path = os.path.join(exp_dir, algo)
            alg_dir_param_name = [name for name in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, name))]
            for algo_param in alg_dir_param_name:
                algo_param_dir = os.path.join(algo_path, algo_param)
                _, ax = plt.subplots(1, 1, figsize=(8, 6))
                data_path_seeds = [f for f in os.listdir(algo_param_dir) if ".pt" in f][:n_seeds]
                for seed, df in enumerate(data_path_seeds):
                    data_path = os.path.join(algo_param_dir, df)
                    with open(data_path, "rb") as _:
                        data = torch.load(data_path, map_location="cpu")
                    #if dim == 1:
                    #    distribution_gif_1D(algo_param_dir, obj, seed, data, ax)
                    if dim == 2:
                        distribution_gif_2D(algo_param_dir, obj, seed, data, ax)
                    else:
                        return None
                        raise Exception("Dimension of the problem should be 2")

def plot_distribution_path(config, n_seeds=1):
    """
    n_seeds: Number of seeds one wants to plot the trajectory
    """
    ### Here make a loop to plot gif to all relevant configurations

    save_path = config["save_dir"]
    exp_name = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
    for experiment in exp_name:
        exp_dir = os.path.join(save_path, experiment)
        alg_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]        
        dim = int(experiment.split("_")[2][-1])
        for algo in tqdm(alg_name, desc="Processing Algorithms..."):
            if (algo in ["quad", "SNES"]) and (dim == 1):
                algo_path = os.path.join(exp_dir, algo)
                _, ax = plt.subplots(1, 1, figsize=(8, 6))
                data_path_seeds = [f for f in os.listdir(algo_path) if ".pt" in f][:n_seeds]
                
                plot_path = os.path.join(algo_path, "path_distribution")
                if not os.path.exists(plot_path):
                    os.mkdir(plot_path)
                
                for seed, df in enumerate(data_path_seeds):
                    data_path = os.path.join(algo_path, df)
                    with open(data_path, "rb") as _:
                        data = torch.load(data_path, map_location="cpu")
                    
                    mu, sigma = data["mu"], data["sigma"]
                    mu, sigma = [float(el) for el in mu], [float(el) for el in sigma]
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    
                    ax.set_xlim(left=-10., right=10.)
                    ax.set_ylim(bottom=0., top=10)
                    ax.scatter(mu, sigma, marker='o')
                    for i in range(len(mu) - 1):
                        ax.arrow(mu[i], sigma[i], mu[i+1] - mu[i], sigma[i+1] - sigma[i], width = 0.01)
                    
                    fig.savefig(os.path.join(plot_path, f"plot{seed}.png"))
                    plt.close()

if __name__ == "__main__":
    plot_distribution_gif("./logs/testfunction/sphere_test")
    #plot_distribution_gif("./logs/testfunction/function_1")