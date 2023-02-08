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
import matplotlib.pyplot as plt
import numpy as np
import random

def run(save_path: str,
        task:str = "test_function",
        bo_kwargs: Optional[Dict[str, Any]] = None,
        problem_kwargs: Optional[Dict[str, Any]] = None,
        ):

    # Set device, dtype, seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    
    #Get experiment settings
    seed = bo_kwargs["seed"]
    BATCH_SIZE = bo_kwargs["batch_size"]
    NUM_RESTARTS = bo_kwargs["num_restarts"]
    RAW_SAMPLES = bo_kwargs["raw_samples"]
    N_TRIALS = bo_kwargs["n_trials"]
    N_BATCH = bo_kwargs["n_iter"]
    MC_SAMPLES = bo_kwargs["mc_samples"]
    BETA, VAR_PRIOR = bo_kwargs["beta"], bo_kwargs["var_prior"]

    #Set seed and device
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Get problem
    problem_kwargs = problem_kwargs or {}
    problem = get_problem(label=task, device=device, dtype=dtype, problem_kwargs=problem_kwargs)
    objective = problem.objective

    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    #Pi bo parameters
    mean, loc = torch.zeros(problem.dim, device=device, dtype=dtype), VAR_PRIOR*torch.eye(problem.dim, device=device, dtype=dtype)
    pi_distrib = MultivariateNormal(mean, loc)

    verbose = False

    best_observed_all_ei, best_observed_all_pi, best_random_all = [], [], []
    train_yvar = torch.tensor(objective.noise_std**2, device=device, dtype=dtype)
        
    def initialize_model(train_x, train_obj, state_dict=None):
        # define models for objective and constraint
        #model_obj = SingleTaskGP(train_x, train_obj).to(train_x)
        model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
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
    
    def update_random_observations(best_random):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = torch.rand(BATCH_SIZE, 2)
        next_random_best = objective(rand_x).max().item()
        best_random.append(max(best_random[-1], next_random_best))       
        return best_random
    
    def update_random_observations_discrete(best_random):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        n_data = problem.raw_data.shape[0]
        indice = torch.tensor(random.sample(range(n_data), BATCH_SIZE))
        rand_x = torch.tensor(problem.scaled_data[indice, :problem.dim], device=device, dtype=dtype)
        next_random_best = objective(rand_x).max().item()
        best_random.append(max(best_random[-1], next_random_best))       
        return best_random


    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_pi, best_random = [], [], []
        
        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei = problem.generate_initial_data(n=10)
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)

        train_x_pi, train_obj_pi = train_x_ei, train_obj_ei
        best_observed_value_pi = best_observed_value_ei
        mll_pi, model_pi = initialize_model(train_x_pi, train_obj_pi)

        best_observed_ei.append(best_observed_value_ei)
        best_observed_pi.append(best_observed_value_pi)
        best_random.append(best_observed_value_ei)
        
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):    
            
            t0 = time.monotonic()
            
            # fit the models
            fit_gpytorch_mll(mll_ei)
            fit_gpytorch_mll(mll_pi)
            
            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            # for best_f, we use the best observed noisy values as an approximation
            qEI = qExpectedImprovement(
                model=model_ei, 
                best_f=train_obj_ei.max(),
                sampler=qmc_sampler
            )
            
            qPI = piqExpectedImprovement(
                model=model_pi,
                best_f=train_obj_pi.max(),
                pi_distrib=pi_distrib,
                n_iter=iteration,
                beta=BETA,
                sampler=qmc_sampler
            )

            # optimize and get new observation
            if task == "test_function":
                new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(qEI)
                new_x_pi, new_obj_pi = optimize_acqf_and_get_observation(qPI)
            elif task == "airfoil":
                new_x_ei, new_obj_ei = optimize_acqf_and_get_observation_discrete(qEI)
                new_x_pi, new_obj_pi = optimize_acqf_and_get_observation_discrete(qPI)

                    
            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

            train_x_pi = torch.cat([train_x_pi, new_x_pi])
            train_obj_pi = torch.cat([train_obj_pi, new_obj_pi])

            # update progress
            if task == "test_function":
                best_random = update_random_observations(best_random)
            elif task == "airfoil":
                best_random = update_random_observations_discrete(best_random)

            best_value_ei = train_obj_ei.max().item()
            best_observed_ei.append(best_value_ei)

            best_value_pi = train_obj_pi.max().item()
            best_observed_pi.append(best_value_pi)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei, 
                train_obj_ei, 
                model_ei.state_dict(),
            )

            mll_pi, model_pi = initialize_model(
                train_x_pi, 
                train_obj_pi,  
                model_pi.state_dict(),
            )
            
            t1 = time.monotonic()
            
            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI) = "
                    f"time = {t1-t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")
    
        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_pi.append(best_observed_pi)
        best_random_all.append(best_random)
    
    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

    GLOBAL_MAXIMUM = objective.best_value

    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    y_ei = np.asarray(best_observed_all_ei)
    y_pi = np.asarray(best_observed_all_pi)
    y_rnd = np.asarray(best_random_all)

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
    
    ax.plot([0, N_BATCH * BATCH_SIZE], [GLOBAL_MAXIMUM] * 2, 'k', label="true best objective", linewidth=2)
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
    ax.legend(loc="lower right")
    
    plt.savefig(os.path.join(save_path, f"seed-{str(seed).zfill(4)}_Beta-{BETA}_VarPrior-{VAR_PRIOR}_Noise-{objective.noise_std}_Dim-{problem.dim}.pdf"))
    plt.savefig(os.path.join(save_path, f"seed-{str(seed).zfill(4)}_Beta-{BETA}_VarPrior-{VAR_PRIOR}_Noise-{objective.noise_std}_Dim-{problem.dim}.png"))
