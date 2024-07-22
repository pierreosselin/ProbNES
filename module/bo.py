import os
import torch
from .objective import get_objective
from typing import Optional, Any, Dict
from botorch.exceptions import BadInitialCandidatesWarning
import warnings
from tqdm import tqdm
import numpy as np
from .optimizers import load_optimizer
import wandb


### Simplify structure with objecive, problem loading, input scaling or not and step optimizer.
def run(save_path: str,
        problem_name:str = "test_function",
        seed:int = 0,
        verbose_synthesis:int = 0,
        exp_kwargs: Optional[Dict[str, Any]] = None,
        alg_kwargs: Optional[Dict[str, Any]] = None,
        problem_kwargs: Optional[Dict[str, Any]] = None,
        ):

    # Get Algorithm
    label = alg_kwargs["algorithm"]

    # Set device, dtype, seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    torch.set_default_dtype(dtype)
    
    #Get experiment settings
    N_BATCH = exp_kwargs["n_iter"]
    N_INIT = exp_kwargs["n_init"]

    #Set seed and device
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Wandb
    if verbose_synthesis:
        wandb.init(
            # set the wandb project where this run will be logged
            project="bo domain informed",

            # track hyperparameters and run metadata
            config={
            "function": "Ackley",
            "Dim": "1D"
            }
        )

    plot_path = os.path.join(save_path, "plots")
    if verbose_synthesis:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
    
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    #Get problem
    problem_kwargs = problem_kwargs or {}
    objective = get_objective(label=problem_name, device=device, dtype=dtype, problem_kwargs=problem_kwargs)
    
    #Get Optimizer
    optimizer = load_optimizer(label, N_INIT, objective, alg_kwargs, plot_path)
    if verbose_synthesis:
        d_plot = optimizer.plot_synthesis()
        if problem_name == "latent_space":
            ## take mean point and decode
            if label == "ES":
                if alg_kwargs[label]["type"] == "CMAES":
                    mu = optimizer.searcher._get_center()
                else:
                    mu = optimizer.searcher._get_mu()
            elif label == "probES":
                mu = optimizer.distribution.loc
            elif label == "qEI":
                mu = optimizer.train_X[-1]
            else:
                raise NotImplementedError
            image_latent = objective.decode(mu.view(1, -1)).squeeze().cpu()    
            d_plot["decoded image"] = wandb.Image(image_latent)
        wandb.log(d_plot)
                

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for _ in tqdm(range(1, N_BATCH + 1), position=0, leave=True, desc = f"Processing algorithm {label} at seed {seed}"):
        optimizer.step()
        if verbose_synthesis:
            d_plot = optimizer.plot_synthesis()
            if problem_name == "latent_space":
            ## take mean point and decode
                if label == "ES":
                    if alg_kwargs[label]["type"] == "CMAES":
                        mu = optimizer.searcher._get_center()
                    else:
                        mu = optimizer.searcher._get_mu()
                elif label == "probES":
                    mu = optimizer.distribution.loc
                elif label == "qEI":
                    mu = optimizer.train_X[-1]
                else:
                    raise NotImplementedError
                image_latent = objective.decode(mu.view(1, -1)).squeeze().cpu()    
                d_plot["decoded image"] = wandb.Image(image_latent)
            wandb.log(d_plot)

    history_params = torch.vstack(optimizer.params_history_list).cpu()
    history_values = torch.vstack(optimizer.values_history).cpu()
    ### Here take best values by batch / initial values
    #indexes = (torch.arange(N_BATCH+1) * alg_kwargs["batch_size"] + N_INIT - 1).cpu()
    #best_observed = torch.cummax(history_values.flatten(), dim=0).values[indexes]
    output_dict = {
        "label": label,
        "X": history_params,
        "Y": history_values.flatten(),
        "best_value": objective.best_value,
        "N_BATCH": N_BATCH,
        "BATCH_SIZE": alg_kwargs["batch_size"],
        "N_INIT": N_INIT,
        "bounds": problem_kwargs["initial_bounds"]
    }

    if label in ["SNES", "quad"]:
        output_dict["mu"] = optimizer.list_mu
        output_dict["sigma"] = optimizer.list_covar
    
    with open(os.path.join(save_path, f"seed-{str(seed).zfill(4)}.pt"), "wb") as fp:
        torch.save(output_dict, fp)