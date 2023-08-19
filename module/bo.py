import os
import torch
from .base_problem import get_problem
from typing import Optional, Any, Union, Tuple, Callable, Dict
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from .acquisition import piqExpectedImprovement
from torch.distributions.multivariate_normal import MultivariateNormal
from .search import NESWSABI
import warnings
from tqdm import tqdm
import numpy as np
from botorch.utils.transforms import standardize, normalize, unnormalize
from evotorch.algorithms import SNES
from evotorch import Problem
from .quadrature import QuadratureExplorationBis, Quadrature
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from .plot_script import plot_GP_fit, plot_synthesis


### By default the Optimization procedure maximize the objective function (acquisition function maximize)
LIST_LABEL = ["random", "SNES", "piqEI", "quad", "qEI"]

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
    
    #Get experiment settings
    BATCH_SIZE = exp_kwargs["batch_size"]
    N_BATCH = exp_kwargs["n_iter"]
    N_INIT = exp_kwargs["n_init"]

    # Algorithm setting
    NUM_RESTARTS = bo_kwargs["num_restarts"]
    RAW_SAMPLES = bo_kwargs["raw_samples"]
    MC_SAMPLES = bo_kwargs["mc_samples"]
    ACQUISITION_BATCH_OPTIMIZATION = bo_kwargs["batch_acq"]
    NORMALIZE = False
    STANDARDIZE_LABEL = True
    VERBOSE = bo_kwargs["verbose"]
    CANDIDATES_VR = bo_kwargs["candidates_vr"]
    POLICY = bo_kwargs["policy"]
    POLICY_KWARGS = bo_kwargs["policy_setting"]
    #Set seed and device
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Get problem
    problem_kwargs = problem_kwargs or {}
    problem = get_problem(label=problem_name, device=device, dtype=dtype, problem_kwargs=problem_kwargs)
    objective = problem.objective
    
    # SNES 
    if label in ["SNES", "NESWSABI"]:
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
        if label == "SNES":
            VAR_PRIOR = bo_kwargs["var_prior"]
            searcher = SNES(problem_ea, popsize=BATCH_SIZE, stdev_init=VAR_PRIOR, center_init=0.)
        elif label == "NESWSABI":
            searcher = NESWSABI(problem_ea, popsize=BATCH_SIZE, stdev_init=problem_kwargs["initial_bounds"], ranking_method=bo_kwargs["ranking_method"], quad_kwargs=bo_kwargs["quadrature"])
        list_mu, list_sigma = [], []
        list_mu.append(searcher._get_mu())
        list_sigma.append(searcher._get_sigma())
    elif label == "piqEI":
        BETA, VAR_PRIOR = bo_kwargs["beta"], bo_kwargs["var_prior"]
        mean, loc = torch.zeros(problem.dim, device=device, dtype=dtype), VAR_PRIOR*torch.eye(problem.dim, device=device, dtype=dtype)
        pi_distrib = MultivariateNormal(mean, loc)
    elif label == "quad": ## Modify initialization to integrate as well
        VAR_PRIOR = bo_kwargs["var_prior"]
        mean, cov_matrix = torch.zeros(problem.dim, device=device, dtype=dtype), VAR_PRIOR*torch.eye(problem.dim, device=device, dtype=dtype)
        params = [mean, torch.diag(cov_matrix)]
        quad_distrib = MultivariateNormal(params[0], torch.diag(params[1]))
        list_mu, list_sigma = [], []
        list_mu.append(quad_distrib.loc)
        list_sigma.append(quad_distrib.covariance_matrix)

    if label not in LIST_LABEL:
        raise Exception(f"Wrong label, must be in {LIST_LABEL}")

    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    verbose = False
    if VERBOSE:
        if not os.path.exists(os.path.join(save_path, "fitgp")):
            os.makedirs(os.path.join(save_path, "fitgp"))

    def initialize_model(train_x, train_obj, state_dict=None):
        # define models for objective and constraint
        if label == "quad":
            covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=train_x.shape[-1],
                    batch_shape=None,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=None,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            model_obj = SingleTaskGP(train_x, train_obj, covar_module=covar_module).to(train_x)
        else:
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
        if NORMALIZE:
            bounds=torch.stack(
                [
                    torch.zeros(problem.dim, dtype=dtype, device=device),
                    torch.ones(problem.dim, dtype=dtype, device=device),
                ])
        else:
            bounds=problem.bounds

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": ACQUISITION_BATCH_OPTIMIZATION, "maxiter": 200},
        )
        # observe new values 
        if NORMALIZE:
            new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
        else:
            new_x = candidates
        exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
        new_obj = exact_obj
        return new_x, new_obj
    
    def optimize_acqf_and_get_observation_random(acq_func, dist):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates = dist.sample(torch.tensor([CANDIDATES_VR, BATCH_SIZE])).to(device = problem.device, dtype = problem.dtype)
        res = acq_func(candidates)
        new_x = candidates[torch.argmax(res)]

        exact_obj = objective(new_x).unsqueeze(-1) # add output dimension
        new_obj = exact_obj
        return new_x, new_obj
    
    def update_random_observations():
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = unnormalize(torch.rand(BATCH_SIZE, problem.dim, device=device, dtype=dtype), bounds=problem.bounds)
        rand_y = objective(rand_x).unsqueeze(-1)
        return rand_x, rand_y

    # average over multiple trials
    best_observed =  []
    
    if label in ["qEI", "piqEI", "random", "quad"]:
        # call helper functions to generate initial training data and initialize model
        train_x, train_obj, best_observed_value = problem.generate_initial_data(n=N_INIT)

    elif label == "SNES":
        searcher.run(1)
        train_x, train_obj = searcher.population.values, searcher.population.evals
        best_observed_value = train_obj.max().item()
        list_mu.append(searcher._get_mu())
        list_sigma.append(searcher._get_sigma())
    
    elif label == "NESWSABI":
        searcher.run(1, train_x=torch.tensor([], device=device, dtype=dtype), train_y=torch.tensor([], device=device, dtype=dtype))
        train_x, train_obj = searcher.population.values.detach().clone(), searcher.population.evals.detach().clone().flatten()
        best_observed_value = train_obj.max().item()
        list_mu.append(searcher._get_mu())
        list_sigma.append(searcher._get_sigma())
    
    if label in ["qEI", "piqEI", "quad"]:
        if NORMALIZE and STANDARDIZE_LABEL:
            mll, model = initialize_model(normalize(train_x, bounds=problem.bounds), standardize(train_obj))
        elif STANDARDIZE_LABEL:
            mll, model = initialize_model(train_x, standardize(train_obj))
        elif NORMALIZE:
            mll, model = initialize_model(normalize(train_x, bounds=problem.bounds), train_obj)
        else:
            mll, model = initialize_model(train_x, train_obj)

    best_observed.append(best_observed_value)
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in tqdm(range(1, N_BATCH + 1), position=0, leave=True, desc = f"Processing algorithm {label} at seed {seed}"):        
        if label in ["qEI", "piqEI"]:
            
            # fit the models
            fit_gpytorch_mll(mll)
            if VERBOSE and problem.dim == 1:
                if ((iteration + 1) % VERBOSE) == 0:
                    plot_GP_fit(model, model.likelihood, train_x, train_obj, obj=objective, lb=-10., up=10., save_path=save_path, iteration=iteration)
            
            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            if label == "qEI":
                # for best_f, we use the best observed noisy values as an approximation
                af = qExpectedImprovement(
                    model=model, 
                    best_f=standardize(train_obj).max(),
                    sampler=qmc_sampler
                )
            if label == "piqEI":
                af = piqExpectedImprovement(
                    model=model,
                    best_f=standardize(train_obj).max(),
                    pi_distrib=pi_distrib,
                    n_iter=iteration,
                    beta=BETA,
                    sampler=qmc_sampler
                )

        elif label == "quad":
            # fit the models
            fit_gpytorch_mll(mll)

            # for best_f, we use the best observed noisy values as an approximation
            # af = QuadratureExploration(
            #     model=model,
            #     distribution=quad_distrib,
            #     batch_acq = ACQUISITION_BATCH_OPTIMIZATION,
            #     )
            af = QuadratureExplorationBis(
                model=model,
                distribution=quad_distrib,
                )

        # optimize and get new observation
        if label in ["qEI", "piqEI"]:
            new_x, new_obj = optimize_acqf_and_get_observation(af)
        elif label == "quad":
            new_x, new_obj = optimize_acqf_and_get_observation_random(af, quad_distrib)
        elif label == "random":
            new_x, new_obj = update_random_observations()
        elif label == "SNES":
            searcher.run(1)
            new_x, new_obj = searcher.population.values, searcher.population.evals
            list_mu.append(searcher._get_mu())
            list_sigma.append(searcher._get_sigma())

        elif label == "NESWSABI":
            #searcher = NESWSABI(problem_ea, popsize=BATCH_SIZE, stdev_init=problem_kwargs["initial_bounds"], ranking_method=bo_kwargs["ranking_method"], quad_kwargs=bo_kwargs["quadrature"])
            searcher.run(1, train_x=train_x, train_y=train_obj)
            new_x, new_obj = searcher.population.values.detach().clone(), searcher.population.evals.detach().clone().flatten()
            list_mu.append(searcher._get_mu())
            list_sigma.append(searcher._get_sigma())

        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])

        best_value = train_obj.max().item()
        best_observed.append(best_value)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        if label in ["qEI", "piqEI", "quad"]:
            if NORMALIZE and STANDARDIZE_LABEL:
                mll, model = initialize_model(
                    normalize(train_x, bounds=problem.bounds), 
                    standardize(train_obj), 
                    model.state_dict(),
                )
            elif STANDARDIZE_LABEL:
                mll, model = initialize_model(
                    train_x, 
                    standardize(train_obj), 
                    model.state_dict(),
                )
            elif NORMALIZE:
                mll, model = initialize_model(
                    normalize(train_x, bounds=problem.bounds), 
                    train_obj, 
                    model.state_dict(),
                )
            else:
                mll, model = initialize_model(
                    train_x, 
                    train_obj, 
                    model.state_dict(),
                )
        if label == "quad":
            quad = Quadrature(model=model, distribution=quad_distrib, policy=POLICY, policy_kwargs=POLICY_KWARGS)
            quad.gradient_direction()
            quad.maximize_step()
            #print(f"Current Epsilon {quad_distrib.covariance_matrix}, optimal step taken {quad.t_max * quad.d_epsilon}, final variance {quad_distrib.covariance_matrix + quad.t_max * quad.d_epsilon}")
            #print(f"Current mu {quad_distrib.loc}, optimal step taken {quad.t_max * quad.d_mu}, final variance {quad_distrib.loc + quad.t_max * quad.d_mu}")
            quad_distrib = quad.update_distribution()
            list_mu.append(quad_distrib.loc.detach().clone())
            list_sigma.append(quad_distrib.covariance_matrix.detach().clone())

            if (verbose_synthesis) and (problem.dim == 1) and (seed == 0):
                if (iteration % verbose_synthesis) == 0:
                    plot_GP_fit(model, model.likelihood, train_x, train_obj, obj=objective, lb=-10., up=10., save_path=save_path, iteration=iteration)
                    if STANDARDIZE_LABEL:
                        std_y, mean_y = torch.std_mean(train_obj)
                        plot_synthesis(model, quad, objective, problem_kwargs["initial_bounds"], iteration, batch_size=BATCH_SIZE, save_path=save_path, standardize=True, mean_Y=float(mean_y), std_Y=float(std_y))
                    else:
                        plot_synthesis(model, quad, objective, problem_kwargs["initial_bounds"], iteration, batch_size=BATCH_SIZE, save_path=save_path)
        
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
        "BATCH_SIZE": BATCH_SIZE,
        "bounds": problem_kwargs["initial_bounds"]
    }

    if label in ["SNES", "NESWSABI", "quad"]:
        output_dict["mu"] = list_mu
        output_dict["sigma"] = list_sigma
    
    with open(os.path.join(save_path, f"seed-{str(seed).zfill(4)}.pt"), "wb") as fp:
        torch.save(output_dict, fp)