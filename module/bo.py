import os
import torch
from .base_problem import get_problem
from typing import Optional, Any, Union, Tuple, Callable, Dict
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from .acquisition import piqExpectedImprovement
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import warnings
from tqdm import tqdm
import numpy as np
import random

from evotorch.algorithms import SNES
from evotorch import Problem

def run(save_path: str,
        problem_name:str = "test_function",
        seed:int = 0,
        exp_kwargs: Optional[Dict[str, Any]] = None,
        bo_kwargs: Optional[Dict[str, Any]] = None,
        problem_kwargs: Optional[Dict[str, Any]] = None,
        ):

    # Get Algorithm
    label = bo_kwargs["algorithm"]

    # Set device, dtype, seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    
    #Get experiment settings
    BATCH_SIZE = exp_kwargs["batch_size"]
    N_BATCH = exp_kwargs["n_iter"]

    # Algorithm setting
    NUM_RESTARTS = bo_kwargs["num_restarts"]
    RAW_SAMPLES = bo_kwargs["raw_samples"]
    MC_SAMPLES = bo_kwargs["mc_samples"]
    BETA, VAR_PRIOR = bo_kwargs["beta"], bo_kwargs["var_prior"]

    #Set seed and device
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Get problem
    problem_kwargs = problem_kwargs or {}
    problem = get_problem(label=problem_name, device=device, dtype=dtype, problem_kwargs=problem_kwargs)
    objective = problem.objective
    
    if label == "SNES":
        # Get Problem for EA
        problem_ea = Problem(
                "max",
                objective,
                initial_bounds=(-problem_kwargs["initial_bounds"], problem_kwargs["initial_bounds"]),
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                solution_length=problem.dim,
                # Evaluation is vectorized
                vectorized=True,
                # Higher-than-default precision
                dtype=torch.float64,
            )
        searcher = SNES(problem_ea, popsize=BATCH_SIZE, stdev_init=1.)
    
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    #Pi bo parameters
    if label == "piqEI":
        mean, loc = torch.zeros(problem.dim, device=device, dtype=dtype), VAR_PRIOR*torch.eye(problem.dim, device=device, dtype=dtype)
        pi_distrib = MultivariateNormal(mean, loc)

    verbose = True

    #train_yvar = torch.tensor(objective.noise_std**2, device=device, dtype=dtype)
        
    def initialize_model(train_x, train_obj, state_dict=None):
        # define models for objective and constraint
        model_obj = SingleTaskGP(train_x, train_obj).to(train_x)
        #model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
        mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
        # load state dict if it is passed
        if state_dict is not None:
            model_obj.load_state_dict(state_dict)
        return mll, model_obj

    def optimize_acqf_and_get_observation(acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=problem.bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values 
        new_x = candidates.detach()
        exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
        new_obj = exact_obj
        return new_x, new_obj
    
    def optimize_acqf_and_get_observation_discrete(acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf_discrete(
            acq_function=acq_func,
            q=BATCH_SIZE,
            choices=torch.tensor(problem.scaled_data[:, :5], device=device, dtype=dtype)
        )
        # observe new values 
        new_x = candidates.detach()
        exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
        new_obj = exact_obj
        return new_x, new_obj
    
    def update_random_observations():
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = 20*torch.rand(BATCH_SIZE, problem.dim, device=device, dtype=dtype) - 10
        rand_y = objective(rand_x).unsqueeze(-1)
        return rand_x, rand_y
    
    def update_random_observations_discrete():
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        n_data = problem.raw_data.shape[0]
        indice = torch.tensor(random.sample(range(n_data), BATCH_SIZE))
        rand_x = torch.tensor(problem.scaled_data[indice, :problem.dim], device=device, dtype=dtype)
        rand_y = objective(rand_x).unsqueeze(-1)      
        return rand_x, rand_y

    # average over multiple trials    
    best_observed =  []
    
    if label in ["qEI", "piqEI", "random"]:
        # call helper functions to generate initial training data and initialize model
        train_x, train_obj, best_observed_value = problem.generate_initial_data(n=10)
    elif label == "SNES":
        searcher.run(1)
        train_x, train_obj = searcher.population.values, searcher.population.evals
        best_observed_value = train_obj.max().item()
    
    if label in ["qEI", "piqEI"]:
        mll, model = initialize_model(train_x, train_obj)

    best_observed.append(best_observed_value)
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in tqdm(range(1, N_BATCH + 1), position=0, leave=True, desc = f"Processing algorithm {label} at seed {seed}"):
        
        t0 = time.monotonic()
        
        if label in ["qEI", "piqEI"]:

            # fit the models
            fit_gpytorch_mll(mll)
            
            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            if label == "qEI":
                # for best_f, we use the best observed noisy values as an approximation
                af = qExpectedImprovement(
                    model=model, 
                    best_f=train_obj.max(),
                    sampler=qmc_sampler
                )
            if label == "piqEI":                
                af = piqExpectedImprovement(
                    model=model,
                    best_f=train_obj.max(),
                    pi_distrib=pi_distrib,
                    n_iter=iteration,
                    beta=BETA,
                    sampler=qmc_sampler
                )

        # optimize and get new observation
        if label in ["qEI", "piqEI"]:
            if problem_name == "test_function":
                new_x, new_obj = optimize_acqf_and_get_observation(af)
            elif problem_name == "airfoil":
                new_x, new_obj = optimize_acqf_and_get_observation_discrete(af)
        elif label == "random":
            if problem_name == "test_function":
                new_x, new_obj = update_random_observations()
            elif problem_name == "airfoil":
                new_x, new_obj = update_random_observations_discrete()
        elif label == "SNES":
            searcher.run(1)
            new_x, new_obj = searcher.population.values, searcher.population.evals
        
        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])

        best_value = train_obj.max().item()
        best_observed.append(best_value)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        if label in ["qEI", "piqEI"]:
            mll, model = initialize_model(
                train_x, 
                train_obj, 
                model.state_dict(),
            )
        
        t1 = time.monotonic()
        
        if verbose:
            if (iteration + 1)%10 == 0:
                print(
                    f"Iteration {iteration} and sample points {new_x} and best observed {best_value}"
                )
        else:
            print(".", end="")

    best_observed = torch.tensor(best_observed, dtype=dtype)
    regret = objective.best_value - best_observed
    output_dict = {
        "label": label,
        "X": train_x.cpu(),
        "Y": best_observed.cpu(),
        "regret": regret.cpu(),
        "N_BATCH": N_BATCH,
        "BATCH_SIZE": BATCH_SIZE
    }
    
    with open(os.path.join(save_path, f"seed-{str(seed).zfill(4)}_Beta-{BETA}_VarPrior-{VAR_PRIOR}_Noise-{objective.noise_std}_Dim-{problem.dim}.pt"), "wb") as fp:
        torch.save(output_dict, fp)
    """
     output_dict = {
        "label": problem_name,
        "X": train_x_ei.cpu(),
        "Y": best_observed_all_ei.cpu(),
        "wall_time": wall_time,
        "best_obj": best_obj,
        "regret": regret,
    }
    
    with open(os.path.join(save_path, f"seed-{str(seed).zfill(4)}_Beta-{BETA}_VarPrior-{VAR_PRIOR}_Noise-{objective.noise_std}_Dim-{problem.dim}.pt"), "wb") as fp:
        torch.save(output_dict, fp)
    """
    """
    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

    GLOBAL_MAXIMUM = objective.best_value

    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    y_ei = np.asarray(best_observed_all_ei)
    y_pi = np.asarray(best_observed_all_pi)
    y_rnd = np.asarray(best_random_all)
    y_snes = np.asarray(best_random_all)

    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(iters, y_ei.mean(axis=0), ".-", label="qEI", color="#1f77b4")
    yerr=ci(y_ei)
    ax.fill_between(iters, y_ei.mean(axis=0)-yerr, y_ei.mean(axis=0)+yerr, alpha=0.1, color="#1f77b4")

    ax.plot(iters, y_pi.mean(axis=0), ".-", label="pi_qEI", color="#8c564b")
    yerr=ci(y_pi)
    ax.fill_between(iters, y_pi.mean(axis=0)-yerr, y_pi.mean(axis=0)+yerr, alpha=0.1, color="#8c564b")

    ax.plot(iters, y_rnd.mean(axis=0), ".-", label="random", color="#ff7f0e")
    yerr=ci(y_rnd)
    ax.fill_between(iters, y_rnd.mean(axis=0)-yerr, y_rnd.mean(axis=0)+yerr, alpha=0.1, color="#ff7f0e")

    ax.plot(iters, y_snes.mean(axis=0), ".-", label="SNES", color="#2ca02c")
    yerr=ci(y_snes)
    ax.fill_between(iters, y_snes.mean(axis=0)-yerr, y_snes.mean(axis=0)+yerr, alpha=0.1, color="#2ca02c")
    
    ax.plot([0, N_BATCH * BATCH_SIZE], [GLOBAL_MAXIMUM] * 2, 'k', label="true best objective", linewidth=2)
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
    ax.legend(loc="lower right")
    
    plt.savefig(os.path.join(save_path, label, f"seed-{str(seed).zfill(4)}_Beta-{BETA}_VarPrior-{VAR_PRIOR}_Noise-{objective.noise_std}_Dim-{problem.dim}.pdf"))
    plt.savefig(os.path.join(save_path, label,f"seed-{str(seed).zfill(4)}_Beta-{BETA}_VarPrior-{VAR_PRIOR}_Noise-{objective.noise_std}_Dim-{problem.dim}.png"))
    """