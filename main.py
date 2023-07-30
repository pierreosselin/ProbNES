from module.bo import run
from module.plot_script import plot_figure, plot_distribution_gif, plot_distribution_path
import os
import argparse
import yaml
from itertools import product


def create_path(save_path, problem_name, problem_kwargs, bo_kwargs):
    
    if problem_name == "test_function":
        s = "_".join([problem_kwargs["function"], f'noise-{problem_kwargs["noise"]}', f'dim-{problem_kwargs["dim"]}', f'initial_bounds-{problem_kwargs["initial_bounds"]}',
            f'beta-{bo_kwargs["beta"]}', f'var_prior-{bo_kwargs["var_prior"]}'])
    elif problem_name == "airfoil":
        raise NotImplementedError
        s = "_".join([f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}'])
    save_path = os.path.join(save_dir, s)
    return save_path


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
    bo_kwargs = config["bo_settings"]
    gpu_label = config["gpu"]
    verbose_synthesis = config["verbose_synthesis"]
    if gpu_label:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_label)

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

            #### Build new save dir ba_m-3_beta-0.01_gamma-0.005_n-5000_epsilon-5e-4_iter-100_abs
            exp_path = create_path(save_dir, problem_name, problem_kwargs, bo_kwargs)
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                print("Processing", exp_path, "...")
            else:
                "If folder already exists then perform optimization depending on OVERWRITE"
                if OVERWRITE == False:
                    print(exp_path + "found without overwriting, next config...")
                    continue

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
                save_path = os.path.join(exp_path, algo)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                initial_seed = config["seed"]
                run(save_path=save_path,
                    problem_name=problem_name,
                    seed=initial_seed + seed,
                    verbose_synthesis=verbose_synthesis,
                    exp_kwargs=exp_kwargs,
                    bo_kwargs=bo_kwargs,
                    problem_kwargs=problem_kwargs,
                    )
    else:
        exp_path = create_path(save_dir, problem_name, problem_kwargs, bo_kwargs)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            print("Processing", exp_path, "...")
        else:
            "If folder already exists then perform optimization depending on OVERWRITE"
            if OVERWRITE == False:
                print(exp_path + "found without overwriting, next config...")
        
        for seed in range(exp_kwargs["n_exp"]):
            label = bo_kwargs["algorithm"]
            save_path = os.path.join(save_dir, label)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            algo = bo_kwargs["algorithm"]
            save_path = os.path.join(exp_path, algo)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            initial_seed = config["seed"]
            run(save_path=save_path,
                problem_name=problem_name,
                seed = initial_seed + seed,
                verbose_synthesis=verbose_synthesis,
                exp_kwargs=exp_kwargs,
                bo_kwargs=bo_kwargs,
                problem_kwargs=problem_kwargs,
                )
    
    plot_figure(save_dir)
    plot_figure(save_dir, log_transform=True)

    ## config["test_function"]
    plot_distribution_gif(config, n_seeds=1)
    
    plot_distribution_path(config, n_seeds=1)

    
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