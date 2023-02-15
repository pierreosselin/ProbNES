from module.bo import run
from module.plot_script import plot_figure, plot_distribution_gif
import os
import argparse
import yaml
from itertools import product

## TODO Manage option for gpu
if __name__ == "__main__":

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
    bo_kwargs = config["bo_settings"]

    ### Make lists for multiple experiments
    list_keys, list_values = [], []
    for key, value in problem_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["pb", key]))
            list_values.append(value)
    for key, value in bo_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["bo", key]))
            list_values.append(value)
    for key, value in exp_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["exp", key]))
            list_values.append(value)
    
    if len(list_values) > 0:
        for t in product(*list_values):
            for i, el in enumerate(t):
                type_param, key = list_keys[i]
                if type_param == "pb":
                    problem_kwargs[key] = el
                elif type_param == "bo":
                    bo_kwargs[key] = el
                elif type_param == "exp":
                    exp_kwargs[key] = el

            for seed in range(exp_kwargs["n_exp"]):
                """
                try:
                    algo = bo_kwargs["algorithm"]
                    save_path = os.path.join(save_dir, algo)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    initial_seed = config["seed"]
                    run(save_path=save_path,
                        problem_name=problem_name,
                        seed = initial_seed + seed,
                        exp_kwargs=exp_kwargs,
                        bo_kwargs=bo_kwargs,
                        problem_kwargs=problem_kwargs,
                        )
                except:
                    print(f"Run failed at parameters {t}, proceeding to the next parameters...")
                    continue
                """
                algo = bo_kwargs["algorithm"]
                save_path = os.path.join(save_dir, algo)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                initial_seed = config["seed"]
                run(save_path=save_path,
                    problem_name=problem_name,
                    seed = initial_seed + seed,
                    exp_kwargs=exp_kwargs,
                    bo_kwargs=bo_kwargs,
                    problem_kwargs=problem_kwargs,
                    )
    else:
        for seed in range(exp_kwargs["n_exp"]):
            label = bo_kwargs["algorithm"]
            save_path = os.path.join(save_dir, label)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            initial_seed = config["seed"]
            run(save_path=save_path,
                problem_name=problem_name,
                seed = initial_seed + seed,
                exp_kwargs=exp_kwargs,
                bo_kwargs=bo_kwargs,
                problem_kwargs=problem_kwargs,
                )
    plot_figure(os.path.dirname(save_path))
    plot_distribution_gif(save_dir, n_seeds=1)

    
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