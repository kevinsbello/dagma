import os, sys, subprocess, random
import numpy as np


# NUM_NODES = [20, 40, 60, 80, 100]
NUM_NODES = [20, 30, 50, 80, 100]
# NUM_NODES = [20]
# NUM_NODES = [500] 
NUM_REPS = 10

# graph params
n = 5000
s0 = 4
graph_type = 'ER'
sem_type = 'gauss'
method = 'dagma'

# model params

# loss_type = 'logistic'
# lambda1, w_threshold = 0.007, 0.3
# max_iter, warm_iter = 6e4, 3e4
# rho_iter, rho_init = 5, 10
# mu_factor = 0.1
# lr = 0.0003
# alpha = '1 .9 .8 .7 .6'
# verbose = True
# checkpoint = 1000

loss_type = 'l2'
lambda1, w_threshold = 0.03, 0.3
max_iter, warm_iter = 6e4, 3e4
rho_iter, rho_init = 5, 1
mu_factor = 0.1
lr = 0.0003
alpha = '1 .9 .8 .7 .6'
verbose = True
checkpoint = 1000



# filepath = '/home/bellok/dagma/csv/neurips_logistic_dagma.csv'
filepath = '/home/bellok/dagma/csv/neurips_linear_dagma.csv'




SEED = 1 # for large 1337, for small 1
random.seed(SEED)
np.random.seed(SEED)
SEEDS = np.random.randint(5000, size=NUM_REPS)
# SEEDS = [4225]

job_directory = f"/home/bellok/dagma"
job_file = os.path.join(job_directory, "runner.sh")
for d in NUM_NODES:
# for n in NUM_SAMPLES:
    for it in range(NUM_REPS):
        seed = SEEDS[it]
        job_name = f'{method}_{graph_type}{s0}_{sem_type}_d={d}_seed={seed}'
        
        args1 = f"--seed {seed} --gt {graph_type} --st {sem_type} --d {d} --s0 {s0} --n {n}"
        args2 = f"--method {method} --l1 {lambda1} --lr {lr} --loss_type {loss_type} --checkpoint {checkpoint}"
        args3 = f"--rho_init {rho_init} --mu_factor {mu_factor} --rho_iter {rho_iter} --alpha {alpha} --w_t {w_threshold} --max_iter {max_iter}"
        args4 = f"--warm_iter {warm_iter} --verbose {verbose} --filepath {filepath} --jobid $SLURM_JOB_ID"
        
        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name='{job_name}'\n")
            fh.writelines(f"#SBATCH --output='{job_directory}/.out/%A_{job_name}.out'\n")
            # fh.writelines(f"#SBATCH --error='{job_directory}/.out/%A_{job_name}.err'\n")
            
            # FOR MERCURY
            fh.writelines("#SBATCH --account=pi-naragam\n") 
            fh.writelines("#SBATCH --partition=standard\n")
            fh.writelines("#SBATCH --nodes=1\n")
            fh.writelines("#SBATCH --cpus-per-task=8\n")
            fh.writelines("#SBATCH --mem-per-cpu=4G\n")
            fh.writelines("#SBATCH --time=1:00:00\n")
            fh.writelines("\nunset XDG_RUNTIME_DIR\n")
            fh.writelines("module load anaconda/2021.05\n")
            fh.writelines("conda init bash\n")
            fh.writelines("source ~/.bashrc\n")
            fh.writelines("conda activate NOTEARS_PLUSPLUS\n")
            fh.writelines(f"srun --unbuffered python /home/bellok/dagma/method_runner.py {args1} {args2} {args3} {args4}\n")
            
            # FOR RCC
            # fh.writelines("#SBATCH --partition=broadwl\n")
            # fh.writelines("#SBATCH --nodes=1\n")
            # fh.writelines("#SBATCH --cpus-per-task=8\n")
            # fh.writelines("#SBATCH --mem-per-cpu=4G\n")
            # fh.writelines("#SBATCH --time=10:00:00\n")
            # fh.writelines("\nunset XDG_RUNTIME_DIR\n")
            # fh.writelines("module load java\n")
            # fh.writelines("module load python/anaconda-2021.05\n")
            # fh.writelines("\nsource activate ~/envs/NOTEARS_PLUSPLUS\n")
            # fh.writelines(f"srun --unbuffered python3 /home/bellok/notears_plusplus/dagma/method_runner.py {args1} {args2} {args3} {args4}\n")

        return_code = subprocess.run(f"sbatch {job_file}", shell=True)
        print(f"d={d} -- it={it} -- {job_name} -- {return_code}")