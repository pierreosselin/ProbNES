from module.bo import run
from module.plot_script import plot_figure, plot_config, plot_distribution_gif, plot_distribution_path
from module.utils import create_path_exp, create_path_alg
import os
import argparse
import yaml
from itertools import product
import torch

if __name__ == "__main__":
    OVERWRITE = True
    # parse arguments and load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test')
    args = parser.parse_args()
    with open(f'config/{args.config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    ### Place where design save_path from config parameters
    save_dir = config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ### Get different  configs
    problem_name=config["problem_name"]
    exp_kwargs = config["exp_settings"]
    problem_kwargs = config["problem_settings"]
    alg_kwargs = config["alg_settings"]
    gpu_label = config["gpu"]
    verbose_synthesis = config["verbose_synthesis"]
    #if gpu_label != 'cpu':
    #    torch.set_default_device('cuda:'+str(gpu_label))

    ### Make lists for multiple experiments
    list_keys, list_values = [], []
    for key, value in problem_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["pb", key]))
            list_values.append(value)
    
    for key, value in alg_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["alg", key]))
            list_values.append(value)

    for key, value in exp_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["exp", key]))
            list_values.append(value)
    
    if type(alg_kwargs["algorithm"]) == list:
        list_algos = alg_kwargs["algorithm"]
    else:
        list_algos = [alg_kwargs["algorithm"]]
    
    dict_keys_algo = {}
    for algo in list_algos:
        list_keys_algo, list_values_algo = [], []
        for key, value in alg_kwargs[algo].items():
            if type(value) == list:
                list_keys_algo.append(key)
                list_values_algo.append(value)
        dict_keys_algo[algo] = tuple([list_keys_algo, list_values_algo])

    for t in product(*list_values): ## For loop on experiment problem parameters and algorithms
        for i, el in enumerate(t):
            type_param, key = list_keys[i]
            if type_param == "pb":
                problem_kwargs[key] = el
            elif type_param == "alg":
                alg_kwargs[key] = el
            elif type_param == "exp":
                exp_kwargs[key] = el
        
        ## Loop on algorithm configurations
        list_keys_algo, list_values_algo = dict_keys_algo[alg_kwargs["algorithm"]]
        for t_algo in product(*list_values_algo):
            for i, el in enumerate(t_algo):
                alg_kwargs[alg_kwargs["algorithm"]][list_keys_algo[i]] = el

            exp_path = create_path_exp(save_dir, problem_name, problem_kwargs)
            #### Build new save dir for problem

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)

            algo = alg_kwargs["algorithm"]
            algo_path = os.path.join(exp_path, algo)
            if not os.path.exists(algo_path):
                os.makedirs(algo_path)

            alg_path = create_path_alg(algo_path, algo, alg_kwargs)
            if not os.path.exists(alg_path):
                os.makedirs(alg_path)
            else:
                if OVERWRITE == False:
                    print(alg_path + "found without overwriting, next config...")
                    continue
            print("Processing", alg_path, "...")
            for seed in range(exp_kwargs["n_exp"]):
                """ To uncomment later when debug no longer needed
                try:
                    algo = alg_kwargs["algorithm"]
                    save_path = os.path.join(save_dir, algo)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    initial_seed = config["seed"]
                    run(save_path=save_path,
                        problem_name=problem_name,
                        seed = initial_seed + seed,
                        exp_kwargs=exp_kwargs,
                        alg_kwargs=alg_kwargs,
                        problem_kwargs=problem_kwargs,
                        )
                except:
                    print(f"Run failed at parameters {t}, proceeding to the next parameters...")
                    continue
                """
                initial_seed = config["seed"]
                run(save_path=alg_path,
                    problem_name=problem_name,
                    seed=initial_seed + seed,
                    verbose_synthesis=verbose_synthesis,
                    exp_kwargs=exp_kwargs,
                    alg_kwargs=alg_kwargs,
                    problem_kwargs=problem_kwargs,
                    )
    
    # plot_figure(save_dir)
    # plot_figure(save_dir, log_transform=True)

    ## config["test_function"]
    # plot_distribution_gif(config, n_seeds=1)
    
    # plot_distribution_path(config, n_seeds=1)

    
    """
    for seed, var_prior in product(seed_list, var_list):
        print(f"Experiments for seed {seed} and var_prior {var_prior}")
        run(save_path=save_dir,
            seed=seed,
            var_prior=var_prior,
            task=task,
            problem_kwargs=problem_kwargs,
            )
    """
    """
    IF PARALLELIZATION:

    size_buffer_list = [50, 100]
    N_list = [2, 5, 10]
    T_list = [100, 500, 1000, 10000]
    threshold_list = [0.1, 0.5, 0.7, 1.]
    seed_list = [0, 1, 2]

    paramlist = list(itertools.product(size_buffer_list, N_list, T_list, threshold_list, seed_list))
    list_dict = []

    for buffer, N, T, threshold, seed in paramlist:
        list_dict.append({"size_buffer": buffer, "N": N, "T": T, "threshold": threshold, "seed": seed})
    
    N = mp.cpu_count()
    print('Number of parallelisable cores: ', N)

    with mp.Pool(processes = N) as p:
        p.map(sim, list_dict)
        
    for buffer, N, T, threshold, seed in paramlist:
        list_dict.append({"size_buffer": buffer, "N": N, "T": T, "threshold": threshold, "seed": seed})
    
    N = mp.cpu_count()
    print('Number of parallelisable cores: ', N)

    with mp.Pool(processes = N) as p:
        p.map(main, list_dict)
    """