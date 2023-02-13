import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
import imageio
from tqdm import tqdm


"""
Scripts to produce plots from the files contained in path
""" 


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
    ax.plot(iters, y.mean(axis=0), ".-", label=label)
    yerr=ci(y, N_TRIALS)
    ax.fill_between(iters, y.mean(axis=0)-yerr, y.mean(axis=0)+yerr, alpha=0.1)

def plot_figure(save_path):
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    alg_name = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
    for algo in alg_name:
        alg_dir = os.path.join(save_path, algo)
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
        ax.plot(iters, y.mean(axis=0), ".-", label=label)
        yerr=ci(y, N_TRIALS)
        ax.fill_between(iters, y.mean(axis=0)-yerr, y.mean(axis=0)+yerr, alpha=0.1)
    ax.plot([0, N_BATCH * BATCH_SIZE], [0.] * 2, 'k', label="true best objective", linewidth=2)
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
    ax.set_ylim(0,10.)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, f"plot_regret.pdf"))
    plt.savefig(os.path.join(save_path, f"plot_regret.png"))

def plot_distribution_gif(save_path):
    alg_name = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
    for algo in alg_name:
        algo_path = os.path.join(save_path, algo)
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        data_path_seeds = [f for f in os.listdir(algo_path) if ".pt" in f]
        for seed, df in tqdm(enumerate(data_path_seeds), desc="Processing Seeds..."):
            data_path = os.path.join(algo_path, df)
            with open(data_path, "rb") as _:
                data = torch.load(data_path, map_location="cpu")
            X = data["X"]
            label = data["label"]
            objective = data["objective"]
            bounds = data["bounds"]
            BATCH_SIZE = data["BATCH_SIZE"]
            N_BATCH = data["N_BATCH"]
            if label == "SNES":
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
                    if label == "SNES":
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

if __name__ == "__main__":
    plot_distribution_gif("./logs/testfunction/sphere_test")