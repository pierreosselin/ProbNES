import os
import torch
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.test_functions.synthetic import Ackley
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    NOISE_SE = 0.
    train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)
    obj = Ackley(dim = 2)
    objective = lambda x : -obj(x)
    bounds = torch.tensor([[-10.0] * 2, [10.0] * 2], device=device, dtype=dtype)

    BATCH_SIZE = 3 if not SMOKE_TEST else 2
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 32

    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


    N_TRIALS = 3 if not SMOKE_TEST else 2
    N_BATCH = 20 if not SMOKE_TEST else 2
    MC_SAMPLES = 256 if not SMOKE_TEST else 32

    verbose = False

    best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []

    def generate_initial_data(n=10):
        # generate training data
        train_x = torch.rand(n, 2, device=device, dtype=dtype)
        exact_obj = objective(train_x).unsqueeze(-1)  # add output dimension
        train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

        best_observed_value = train_obj.max().item()
        return train_x, train_obj, best_observed_value
        
    def initialize_model(train_x, train_obj, state_dict=None):
        # define models for objective and constraint
        model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
        # combine into a multi-output GP model
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
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values 
        new_x = candidates.detach()
        exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
        new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        return new_x, new_obj


    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_nei, best_random = [], [], []
        
        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei = generate_initial_data(n=10)
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)
                
        best_observed_ei.append(best_observed_value_ei)
        
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):    
            
            t0 = time.monotonic()
            
            # fit the models
            fit_gpytorch_mll(mll_ei)
            
            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            # for best_f, we use the best observed noisy values as an approximation
            qEI = qExpectedImprovement(
                model=model_ei, 
                best_f=train_obj_ei.max(),
                sampler=qmc_sampler
            )
            
            # optimize and get new observation
            new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(qEI)
                    
            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

            # update progress
            best_value_ei = train_obj_ei.max().item()
            best_observed_ei.append(best_value_ei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei, 
                train_obj_ei, 
                model_ei.state_dict(),
            )
            
            t1 = time.monotonic()
            
            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI) = "
                    f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")
    
        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_nei.append(best_observed_nei)
        best_random_all.append(best_random)
    
    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

    GLOBAL_MAXIMUM = 0.

    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    y_ei = np.asarray(best_observed_all_ei)

    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="qEI", linewidth=1.5)
    ax.plot([0, N_BATCH * BATCH_SIZE], [GLOBAL_MAXIMUM] * 2, 'k', label="true best objective", linewidth=2)
    #ax.set_ylim(bottom=0.5)
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
    ax.legend(loc="lower right")

    plt.savefig("hello.pdf")
    plt.show()