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
from module.utils import nearestPD


"""
Scripts to produce plots from the files contained in path
""" 


def posterior_quad(model, theta, var):
    mean_distrib_test, var_distrib_test = torch.tensor([theta], dtype=torch.float64, device=model.train_inputs[0].device), torch.diag(torch.tensor([var], dtype=torch.float64, device=model.train_inputs[0].device))
    quad_distrib_test = MultivariateNormal(mean_distrib_test, var_distrib_test)
    quad_test = Quadrature(model=model,
            distribution=quad_distrib_test,
            c1 = 0.1,
            c2 = 0.2,
            t_max = 1,
            budget = 50)
    
    quad_test.quadrature()
    return quad_test.m.detach().clone(), quad_test.v.detach().clone()

def plot_synthesis(model, quad_distrib, objective, bounds, iteration, bounds_t = 10., save_path=".", standardize=False, mean_Y=None, std_Y=None):
    
    save_path_gp = os.path.join(save_path, f"fitgp/synthesis_{iteration}.png")
    quad = Quadrature(model=model,
            distribution=quad_distrib,
            c1 = 0.1,
            c2 = 0.2,
            t_max = 50,
            budget = 500)
    quad.quadrature()
    quad.gradient_direction()

    b = np.arange(-float(bounds), float(bounds), 0.2)
    d = np.arange(0, 3, 0.05)[1:]**2
    B, D = np.meshgrid(b, d)
    n, m = b.shape[0], d.shape[0]
    res = torch.stack((torch.tensor(B.flatten()), torch.tensor(D.flatten())), axis = 1).numpy()
    result, result_wolfe = [], []
    for el in tqdm(res):
        post = posterior_quad(model, el[0], el[1])
        result.append(post)
    mean = torch.tensor(result).numpy()[:,0].reshape(m,n)
    t_linspace = torch.linspace(-bounds_t, bounds_t, 200, dtype=quad.train_X.dtype)
    result_wolfe = []
    for t in t_linspace:
        result_wolfe.append(quad.compute_p_wolfe(t))
    wolfe_tensor = torch.tensor(result_wolfe)

    ## Compute gradients at multiple places
    b_grad = np.arange(-float(bounds), float(bounds), 1)
    d_grad = np.arange(0, 3, 0.5)[1:]**2
    B_grad, D_grad = np.meshgrid(b_grad, d_grad)
    #n_grad, m_grad = b_grad.shape[0], d_grad.shape[0]
    res_grad = torch.stack((torch.tensor(B_grad.flatten()), torch.tensor(D_grad.flatten())), axis = 1).numpy()
    result_grad, max_length = [], 0
    for el in tqdm(res_grad):
        mean_distrib_grad, var_distrib_grad = torch.tensor([el[0]], dtype=torch.float64, device=model.train_inputs[0].device), torch.diag(torch.tensor([el[1]], dtype=torch.float64, device=model.train_inputs[0].device))
        quad_distrib_grad = MultivariateNormal(mean_distrib_grad, var_distrib_grad)
        quad_grad = Quadrature(model=model,
                distribution=quad_distrib_grad,
                c1 = 0.1,
                c2 = 0.2,
                t_max = 1,
                budget = 50)
        quad_grad.quadrature()
        quad_grad.gradient_direction()
        mu_grad, epsilon_grad = float(quad_grad.d_mu.detach().clone()), float(quad_grad.d_epsilon.detach().clone())
        max_length = max(max_length, np.sqrt(mu_grad**2 + epsilon_grad**2))
        result_grad.append([mu_grad, epsilon_grad])
        
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plot_GP_fit_ax(axs[0,0], model, quad_distrib, model.train_inputs[0], model.train_targets, objective, standardize=standardize, lb=-float(bounds), up=float(bounds), mean_Y=mean_Y, std_Y=std_Y)

    contour1 = axs[0,1].contourf(B, D, mean)
    # Gradient
    arrow_factor = 5
    for i, el in tqdm(enumerate(res_grad)):
        axs[0, 1].arrow(el[0], el[1], arrow_factor*result_grad[i][0]/max_length, arrow_factor*result_grad[i][1]/max_length, width = 0.1)
    axs[0,1].set_xlabel(r'$\mu$')
    axs[0,1].set_ylabel(r'$\sigma^{2}$')
    axs[0,1].set_title(r"Predictive mean of $g(\theta)$")


    contour4 = axs[1,1].contourf(B, D, mean)
    mu2 = float(quad.distribution.loc + float(t_linspace[np.argmax(wolfe_tensor)])*quad.d_mu)
    Epsilon2 = float(nearestPD(quad.distribution.covariance_matrix + float(t_linspace[np.argmax(wolfe_tensor)])*quad.d_epsilon))

    axs[0,1].plot(t_linspace.numpy(), wolfe_tensor.numpy())

    axs[1,1].scatter([float(quad.distribution.loc)], [float(quad.distribution.covariance_matrix)])
    axs[1,1].scatter([mu2], [Epsilon2])
    axs[1,1].arrow(float(quad.distribution.loc), float(quad.distribution.covariance_matrix), float(t_linspace[np.argmax(wolfe_tensor)]*quad.d_mu), float(t_linspace[np.argmax(wolfe_tensor)]*quad.d_epsilon), width = 0.1)

    fig.colorbar(contour1, ax=axs[0,0])
    fig.colorbar(contour1, ax=axs[0,1])
    fig.colorbar(contour4, ax=axs[1,1])
    fig.savefig(save_path_gp)


def plot_GP_fit(model, likelihood, train_X, targets, obj, lb=-10., up=10., save_path=".", iteration=1):
    """ Plot the figures corresponding to the Gaussian process fit
    """
    save_path_gp = os.path.join(save_path, f"fitgp/{iteration}.png")
    model.eval()
    likelihood.eval()
    test_x = torch.linspace(lb, up, 200, device=train_X.device, dtype=train_X.dtype)
    with torch.no_grad():
        # Make predictions
        predictions = likelihood(model(test_x))
        lower, upper = predictions.confidence_region()
    Y_standard, Y_mean, Y_std = standardize_return(targets)
    value_ = ((obj(test_x.unsqueeze(-1)) - Y_mean)/Y_std).flatten()

    plt.scatter(train_X.cpu().numpy(), Y_standard.cpu().numpy(), color='black', label='Training data')
    plt.plot(test_x.cpu().numpy(), predictions.mean.cpu().numpy(), color='blue', label='Predictive mean')
    plt.plot(test_x.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='lightblue', alpha=0.5, label='Confidence region')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.savefig(save_path_gp)
    plt.clf()


def plot_GP_fit_ax(ax, model, distribution, train_X, targets, obj, standardize=False, lb=-10., up=10., mean_Y=None, std_Y=None):
    """ Plot the figures corresponding to the Gaussian process fit
    """
    model.eval()
    model.likelihood.eval()
    test_x = torch.linspace(lb, up, 200, device=train_X.device, dtype=train_X.dtype)
    with torch.no_grad():
        # Make predictions
        predictions = model.likelihood(model(test_x))
        lower, upper = predictions.confidence_region()
    
    if standardize:
        predictions = predictions*float(std_Y) + float(mean_Y)
        lower, upper = lower*float(std_Y) + float(mean_Y), upper*float(std_Y) + float(mean_Y)
        targets = targets*float(std_Y) + float(mean_Y)
    value_ = (obj(test_x.unsqueeze(-1))).flatten()

    ax.scatter(train_X.cpu().numpy(), targets.cpu().numpy(), color='black', label='Training data')
    ax.plot(test_x.cpu().numpy(), predictions.mean.cpu().numpy(), color='blue', label='Predictive mean')
    ax.plot(test_x.cpu().numpy(), value_.cpu().numpy(), color='green', label='True Function')
    ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='lightblue', alpha=0.5, label='Confidence region')
    
    x = np.linspace((distribution.loc - 3*distribution.covariance_matrix).cpu().detach().numpy(), (distribution.loc + 3*distribution.covariance_matrix).cpu().detach().numpy(), 100).flatten()
    y_lim = ax.get_ylim()
    ax.plot(x, (y_lim[1] - y_lim[0])*stats.norm.pdf(x, distribution.loc.cpu().detach().numpy(), distribution.covariance_matrix.cpu().detach().numpy()).flatten(), "k")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gaussian Process Regression')
    ax.legend()

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
        with open(data_path, "rb") as _:
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
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        alg_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]
        for algo in alg_name:
            alg_dir = os.path.join(exp_dir, algo)
            data_path_seeds = [f for f in os.listdir(alg_dir) if ".pt" in f]
            data_over_seeds = []
            for _, df in enumerate(data_path_seeds):
                data_path = os.path.join(alg_dir, df)
                with open(data_path, "rb") as _:
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
            plt.savefig(os.path.join(exp_dir, f"plot_regret.pdf"))
            plt.savefig(os.path.join(exp_dir, f"plot_regret.png"))
        else:
            plt.savefig(os.path.join(exp_dir, f"plot_regret_log.pdf"))
            plt.savefig(os.path.join(exp_dir, f"plot_regret_log.png"))

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
    nu = -objective(torch.tensor(b).reshape(-1,1)).numpy()
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
        obj = get_objective(config["problem_name"], **{"function": experiment.split("_")[0], "dim":dim, "noise_std":float(experiment.split("_")[1][6:])})
        for algo in tqdm(alg_name, desc="Processing Algorithms..."):
            algo_path = os.path.join(exp_dir, algo)
            _, ax = plt.subplots(1, 1, figsize=(8, 6))
            data_path_seeds = [f for f in os.listdir(algo_path) if ".pt" in f][:n_seeds]
            for seed, df in enumerate(data_path_seeds):
                data_path = os.path.join(algo_path, df)
                with open(data_path, "rb") as _:
                    data = torch.load(data_path, map_location="cpu")
                if dim == 1:
                    distribution_gif_1D(algo_path, obj, seed, data, ax)
                elif dim == 2:
                    distribution_gif_2D(algo_path, obj, seed, data, ax)
                else:
                    raise Exception("Dimension of the problem should be less than 2")

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