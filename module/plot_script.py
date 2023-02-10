import os
import matplotlib.pyplot as plt
import numpy as np
import torch
"""
Scripts to produce plots from the files contained in path
"""   

def ci(y, N_TRIALS):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

def plot_figure(save_path):
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    alg_name = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
    for algo in alg_name:
        alg_dir = os.path.join(save_path, algo)
        data_path_seeds = [f for f in os.listdir(alg_dir) if ".pt" in f]
        data_over_seeds = []
        for _, df in enumerate(data_path_seeds):
            data_path = os.path.join(alg_dir, df)
            with open(data_path, "rb") as fp:
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

if __name__ == "__main__":
    plot_figure("./logs/testfunction/rosenbrocktest_result")