# %%
import sys, os, contextlib
from pathlib import Path
# sys.path.append('/home/bellok/notears_plusplus')
import argparse
import numpy as np
from timeit import default_timer as timer
import utils
from dagma_linear import DAGMA_linear
import torch
import scipy.linalg as sla
import pandas as pd

def test_dagma(X, B_true, W_true, args):
    model = DAGMA_linear(loss_type=args.loss_type, verbose=args.verbose, dtype=np.float64)
    start_time = timer()
    W_est = model.fit(
        X, args.lambda1, w_threshold=args.w_threshold,
        T=args.rho_iter,  mu_init=args.rho_init, mu_factor=args.mu_factor, s= args.alpha, checkpoint=args.checkpoint,
        warm_iter=args.warm_iter, max_iter=args.max_iter, lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2
    )
    stop_time = timer()
    runtime = stop_time - start_time
    print(f'\nfinished in {round(runtime, 2)}s - ', end='')
    acc, w_thr = utils.do_threshold(W_est, B_true, args.w_threshold)
    acc['runtime'] = runtime
    acc['h'] = model.h_final
    acc['score'] = model.score_final
    return acc, w_thr, W_est



if __name__ == '__main__':
    # new_utils.print_machine_info()
    
    # PARSE CL
    parser = argparse.ArgumentParser()
    # ALGORITHM TO RUN
    parser.add_argument('--method', type=str, required=True)
    # GRAPH MODEL
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--graph_type', '--gt', type=str, required=True)
    parser.add_argument('--sem_type', '--st', type=str, required=True)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--s0', type=int, required=True)
    parser.add_argument('--n', type=float)
    parser.add_argument('--noise_scale', type=int, default=1)
    # NOTEARS ARGS
    parser.add_argument('--lambda1', '--l1', type=float)
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--w_threshold', '--w_t', type=float, default=.3)
    # NOTEARSPP ARGS
    parser.add_argument('--solver', type=str)
    parser.add_argument('--beta_1', type=float, default=.99)
    parser.add_argument('--beta_2', type=float, default=.999)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--rho_iter', type=int)
    parser.add_argument('--alpha', type=float, nargs='+')
    parser.add_argument('--rho_init', type=float, default=1.0)
    parser.add_argument('--mu_factor', type=float)
    parser.add_argument('--max_iter', type=float)
    parser.add_argument('--warm_iter', type=float)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--checkpoint', type=int, default=2000)
    # MISC ABOUT RESULTS
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--jobid', type=str)
    
    
    args = parser.parse_args()
    print(args)

    # SET SEED FOR REPRODUCBILITY
    SEED = args.seed
    utils.set_random_seed(SEED)
    torch.manual_seed(SEED)
    print(f'\nRUNNING WITH SEED {SEED}')
    
    # CREATE GROUND-TRUTH AND DATA
    B_true = utils.simulate_dag(args.d, args.s0 * args.d, args.graph_type)
    W_true = utils.simulate_parameter(B_true)
    noise_scale = args.noise_scale
    if noise_scale == -1:
        noise_scale = 0.5 + 3 * np.random.rand(args.d)
    try:
        n = int(args.n)
    except:
        n = args.n
    X = utils.simulate_linear_sem(W_true, n, args.sem_type, noise_scale=noise_scale)
    
    # RUN METHOD
    if args.method == 'dagma':
        print(f'\nmethod = {args.method}')
        acc, w_thr, W_est = test_dagma(X, B_true, W_true, args)
        acc['w_err'] = np.linalg.norm(W_est - W_true, 'fro') / args.d
    print(f'\nw_t = {w_thr}\n{acc}')
    
    # SAVE RESULTS    
    df = pd.DataFrame({
        'seed': [], 'method': [], 'd': [], 'shd': [], 'tpr': [], 'fdr': [], 'fpr': [], 'runtime': [], 'nnz': [],
        'graph_type': [], 'sem_type': [], 'n': [], 'lambda1': [], 'loss_type': [],
        'w_threshold': [], 'solver': [], 'beta_1': [], 'beta_2': [], 'lr': [], 'jobid': [],
        'rho_iter': [], 'rho_init': [], 'mu_factor': [], 'max_iter': [], 'warm_iter': [], 'w_err': [], 
        'h': [], 'score': [],
    })

    acc['method'] = args.method
    acc['d'] = args.d
    acc['graph_type'] = args.graph_type + str(args.s0)
    acc['sem_type'] = args.sem_type
    acc['n'] = n
    acc['seed'] = args.seed
    acc['lambda1'] = args.lambda1
    acc['loss_type'] = args.loss_type
    acc['w_threshold'] = args.w_threshold
    acc['solver'] = args.solver
    acc['beta_1'] = args.beta_1
    acc['beta_2'] = args.beta_2
    acc['lr'] = args.lr
    acc['jobid'] = args.jobid
    acc['rho_iter'] = args.rho_iter
    acc['rho_init'] = args.rho_init
    acc['mu_factor'] = args.mu_factor
    acc['max_iter'] = args.max_iter
    acc['warm_iter'] = args.warm_iter
    
    df = df.append(acc, ignore_index=True) 
    if args.filepath is not None:
        results_file = Path(args.filepath)
        if results_file.is_file():
            result_str = df.to_csv(header=False, index=False)    
            with open(args.filepath, 'a') as ff:
                ff.write(result_str)
        else:
            result_str = df.to_csv(index=False)
            with open(args.filepath, 'x') as ff:
                ff.write(result_str)
        print(f'Saving to {args.filepath}')
    print(df.to_csv(header=False, index=False))