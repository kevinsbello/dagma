# %%
import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from timeit import default_timer as timer
import utils
from scipy.special import expit as sigmoid


class DAGMA_linear:
    
    def __init__(self, loss_type, verbose=False, dtype=np.float64):
        super().__init__()
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.verbose = verbose
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None
            
    def _score(self, W):
        if self.loss_type == 'l2':
            dif = self.Id - W 
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
        elif self.loss_type == 'logistic':
            M = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, M) - self.X * M).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(M) - self.cov
        return loss, G_loss

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * la.inv(M).T * W 
        return h, G_h

    def _func(self, W, mu, s=1.0):
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def printing(self, W, iter, mu, s):
        self.vprint(f'\nInner iteration {iter}')
        obj, score, h = self._func(W, mu, s)
        self.vprint(f'\th(W_est): {h:.4e}')
        self.vprint(f'\tscore(W_est): {score:.4e}')
        self.vprint(f'\tobj(W_est): {obj:.4e}')
        return obj, score, h
    
    def minimize(self, W, mu, max_iter, s, lr=0.0002, lr_decay=False, beta_1=0.99, beta_2=0.999):
        obj_prev = 1e16
        opt_m, opt_v = 0, 0
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- decay: {lr_decay} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')
        
        for iter in range(1, int(max_iter)+1):
            ## COMPUTE GRADIENT
            M = la.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0): # sI - W o W is not an M-matrix
                self.vprint(f'Min entry of M {np.min(M)}')
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, False
                else:
                    W += lr * grad
                    if lr <= 1e-10:
                        return W, True
                    lr *= .5
                    W -= lr * grad
                    M = la.inv(s * self.Id - W * W)
                    self.vprint(f'Learning rate decreased to lr: {lr}')
            
            if self.loss_type == 'l2':
                G_score = -mu * self.cov @ (self.Id - W) 
            else:
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * M.T * W
            
            ## ADAM
            opt_m *= beta_1; opt_m += (1 - beta_1) * Gobj
            opt_v *= beta_2; opt_v += (1 - beta_2) * (Gobj ** 2)
            m_hat = opt_m / (1 - beta_1 ** iter)
            v_hat = opt_v / (1 - beta_2 ** iter)
            grad = m_hat / (np.sqrt(v_hat) + 1e-8)
            
            ## PERFORM STEP
            if lr_decay:
                W -= max(lr / np.sqrt(iter), 0.00001) * grad
            else:
                W -= lr * grad
            
            ## PRINTING
            if self.verbose and (iter % self.checkpoint == 0 or iter == max_iter):
                obj_new, _, _ = self.printing(W, iter, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= 1e-6:
                    break
                obj_prev = obj_new
        return W, True
    
    def fit(self, X, lambda1, w_threshold=0.2, 
            T=5, mu_init=1.0, mu_factor=0.1, 
            s=[1.0, 0.9, 0.8, 0.7], warm_iter=4e4, 
            max_iter=1e5, lr=0.0003, checkpoint=2000,
            beta_1=0.99, beta_2=0.999,
        ):
        
        ## INITALIZING VARIABLES 
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        
        if self.loss_type == 'l2':
            self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        
        if self.verbose:
            h0, _ = self._h(self.W_est)
            score0, _ = self._score(self.W_est)
            self.vprint(f'h(W_0): {h0:.4f}')
            self.vprint(f'score(W_0): {score0:.4f}')
                    
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        ## START DAGMA
        for i in range(T):
            self.vprint(f'\nIteration -- {i+1}:')
            lr_adam, success = lr, False
            inner_iters = max_iter if i == T - 1 else warm_iter
            lr_decay = False
            while success is False:
                W_temp, success = self.minimize(self.W_est.copy(), mu, inner_iters, s[i], 
                                                lr=lr_adam, lr_decay=lr_decay, beta_1=beta_1, beta_2=beta_2)
                if success is False:
                    self.vprint(f'Retrying with larger s and decay')
                    lr_decay = True
                    s[i] += 0.1
            self.W_est = W_temp
            mu *= mu_factor
        
        ## Store final h and score values and Threshold to remove possible False Discoveries
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est


if __name__ == '__main__':
    
    SEED = np.random.randint(5000)
    # SEED = 2142
    utils.set_random_seed(SEED)
    
    n, d = 1000, 3
    s0 = 3
    # s0 = 2
    graph_type = 'SF'
    sem_type = 'gauss'
    noise_scale = 1
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    
    
    W_true = np.array([
        [0, 1.2, 1],
        [0, 0, -0.3],
        [0, 0, 0]
    ])
    B_true = W_true != 0
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale)
    # print(f'Total num of edges in ground-truth DAG: {np.flatnonzero(B_true)}')
    
    ### NEW METHOD
    model = DAGMA_linear(loss_type='l2', verbose=True)
    Wh, runtime = model.fit(
        X, lambda1=0.0, w_threshold=0, W_true=W_true,
        T=5, mu_init=1, mu_factor=0.1, s=[1], checkpoint=2000,
        warm_iter=2e4, max_iter=6e4, lr=0.0001
    )
    
    print(Wh)
    
    # STATS
    acc = utils.do_threshold(Wh, B_true, t=0.3, inc=0.01)
    print(f'\nSEED: {SEED}')
    print('|W_true - W_gd|: ', la.norm(Wh - W_true, 'fro') / d)
    print(f'time: {runtime:.4f}s')
    print('ACC:', acc, '\n\n')

    try:
        print('\nSkeleton acc:')
        acc = utils.count_accuracy(B_true, -1 * (Wh != 0))
        print(acc)
    except:
        print('\n Skeleton failed')
    
    # CHECKING BEST SHD
    min_acc = utils.get_best_shd(Wh, B_true)
    print('\nBEST THRESHOLD:')
    print(min_acc)

    
# %%
