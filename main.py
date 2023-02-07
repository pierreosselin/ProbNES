from module.bo import run
from itertools import product
import os

if __name__ == "__main__":

    ### Place where design save_path from config parameters
    save_dir = "./logs"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    seed_list = [0,1,2]
    var_list = [1.,100.,10000.]
    task="test_function"
    problem_kwargs={"dim":2, "function":"ackley"}

    for seed, var_prior in product(seed_list, var_list):
        print(f"Experiments for seed {seed} and var_prior {var_prior}")
        run(save_path=save_dir,
            seed=seed,
            var_prior=var_prior,
            task=task,
            problem_kwargs=problem_kwargs,
            )

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