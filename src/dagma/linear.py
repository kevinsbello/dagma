import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm
import typing


__all__ = ["DagmaLinear"]

class DagmaLinear:
    """
    A Python object that contains the implementation of DAGMA for linear models using numpy and scipy.
    """
    
    def __init__(self, loss_type: str, verbose: bool = False, dtype: type = np.float64) -> None:
        r"""
        Parameters
        ----------
        loss_type : str
            One of ["l2", "logistic"]. ``l2`` refers to the least squares loss, while ``logistic``
            refers to the logistic loss. For continuous data: use ``l2``. For discrete 0/1 data: use ``logistic``.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.linear.DagmaLinear.fit`. Defaults to ``False``.
        dtype : type, optional
           Defines the float precision, for large number of nodes it is recommened to use ``np.float64``. 
           Defaults to ``np.float64``.
        """
        super().__init__()
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None
            
    def _score(self, W: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value and gradient of the score function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            loss value, and gradient of the loss function
        """
        if self.loss_type == 'l2':
            dif = self.Id - W 
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
        elif self.loss_type == 'logistic':
            R = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
        return loss, G_loss

    def _h(self, W: np.ndarray, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value and gradient of the logdet acyclicity constraint.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            h value, and gradient of h
        """
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W: np.ndarray, mu: float, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value of the penalized objective function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        mu : float
            Weight of the score function.
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            Objective value, and gradient of the objective
        """
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad: np.ndarray, iter: int, beta_1: float, beta_2: float) -> np.ndarray:
        r"""
        Performs one update of Adam.

        Parameters
        ----------
        grad : np.ndarray
            Current gradient of the objective.
        iter : int
            Current iteration number.
        beta_1 : float
            Adam hyperparameter.
        beta_2 : float
            Adam hyperparameter.

        Returns
        -------
        np.ndarray
            Updates the gradient by the Adam method.
        """
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, 
                 W: np.ndarray, 
                 mu: float, 
                 max_iter: int, 
                 s: float, 
                 lr: float, 
                 tol: float = 1e-6, 
                 beta_1: float = 0.99, 
                 beta_2: float = 0.999, 
                 pbar: tqdm = tqdm(),
                 ) -> typing.Tuple[np.ndarray, bool]:        
        r"""
        Solves the optimization problem: 
            .. math::
                \arg\min_{W \in \mathbb{W}^s} \mu \cdot Q(W; \mathbf{X}) + h(W),
        where :math:`Q` is the score function. This problem is solved via (sub)gradient descent, where the initial
        point is `W`.

        Parameters
        ----------
        W : np.ndarray
            Initial point of (sub)gradient descent.
        mu : float
            Weights the score function.
        max_iter : int
            Maximum number of (sub)gradient iterations.
        s : float
            Number that controls the domain of M-matrices.
        lr : float
            Learning rate.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.
        beta_1 : float, optional
            Hyperparamter for Adam. Defaults to 0.99.
        beta_2 : float, optional
            Hyperparamter for Adam. Defaults to 0.999.
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

        Returns
        -------
        typing.Tuple[np.ndarray, bool]
            Returns an adjacency matrix until convergence or `max_iter` is reached.
            A boolean flag is returned to point success of the optimization. This can be False when at any iteration, the current
            W point went outside of the domain of M-matrices.
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')
        mask_inc = np.zeros((self.d, self.d))
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
                mask_exc[self.exc_r, self.exc_c] = 0.
                
        for iter in range(1, max_iter+1):
            ## Compute the (sub)gradient of the objective
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0): # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f'Learning rate decreased to lr: {lr}')
            
            if self.loss_type == 'l2':
                G_score = -mu * self.cov @ (self.Id - W) 
            elif self.loss_type == 'logistic':
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T + mask_inc * np.sign(W)
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            W *= mask_exc
                
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W_est): {h:.4e}')
                self.vprint(f'\tscore(W_est): {score:.4e}')
                self.vprint(f'\tobj(W_est): {obj_new:.4e}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True
    
    def fit(self, 
            X: np.ndarray,
            lambda1: float = 0.03, 
            w_threshold: float = 0.3, 
            T: int = 5,
            mu_init: float = 1.0, 
            mu_factor: float = 0.1, 
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6], 
            warm_iter: int = 3e4, 
            max_iter: int = 6e4, 
            lr: float = 0.0003, 
            checkpoint: int = 1000, 
            beta_1: float = 0.99, 
            beta_2: float = 0.999,
            exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None, 
            include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
        ) -> np.ndarray :
        r"""
        Runs the DAGMA algorithm and returns a weighted adjacency matrix.

        Parameters
        ----------
        X : np.ndarray
            :math:`(n,d)` dataset.
        lambda1 : float
            Coefficient of the L1 penalty. Defaults to 0.03.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold. Defaults to 0.3.
        T : int, optional
            Number of DAGMA iterations. Defaults to 5.
        mu_init : float, optional
            Initial value of :math:`\mu`. Defaults to 1.0.
        mu_factor : float, optional
            Decay factor for :math:`\mu`. Defaults to 0.1.
        s : typing.Union[typing.List[float], float], optional
            Controls the domain of M-matrices. Defaults to [1.0, .9, .8, .7, .6].
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.linear.DagmaLinear.minimize` for :math:`t < T`. Defaults to 3e4.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.linear.DagmaLinear.minimize` for :math:`t = T`. Defaults to 6e4.
        lr : float, optional
            Learning rate. Defaults to 0.0003.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations. Defaults to 1000.
        beta_1 : float, optional
            Adam hyperparameter. Defaults to 0.99.
        beta_2 : float, optional
            Adam hyperparameter. Defaults to 0.999.
        exclude_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be excluded from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
        include_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be included from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        
        .. important::

            If the output of :py:meth:`~dagma.linear.DagmaLinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        
        .. warning::
            
            While DAGMA ensures to exclude the edges given in ``exclude_edges``, the current implementation does not guarantee that all edges
            in ``included edges`` will be part of the final DAG.
        """ 
        
        ## INITALIZING VARIABLES 
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        
        if self.loss_type == 'l2':
            self.X -= X.mean(axis=0, keepdims=True)
        
        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None
        
        if exclude_edges is not None:
            if type(exclude_edges) is tuple and type(exclude_edges[0]) is tuple and np.all(np.array([len(e) for e in exclude_edges]) == 2):
                self.exc_r, self.exc_c = zip(*exclude_edges)
            else:
                ValueError("blacklist should be a tuple of edges, e.g., ((1,2), (2,3))")
        
        if include_edges is not None:
            if type(include_edges) is tuple and type(include_edges[0]) is tuple and np.all(np.array([len(e) for e in include_edges]) == 2):
                self.inc_r, self.inc_c = zip(*include_edges)
            else:
                ValueError("whitelist should be a tuple of edges, e.g., ((1,2), (2,3))")        
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
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
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nIteration -- {i+1}:')
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(self.W_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        self.vprint(f'Retrying with larger s')
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                mu *= mu_factor
        
        ## Store final h and score values and threshold
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est


if __name__ == '__main__':
    import utils
    from timeit import default_timer as timer
    utils.set_random_seed(1)
    
    n, d, s0 = 500, 20, 20 # the ground truth is a DAG of 20 nodes and 20 edges in expectation
    graph_type, sem_type = 'ER', 'gauss'
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    
    model = DagmaLinear(loss_type='l2')
    start = timer()
    W_est = model.fit(X, lambda1=0.02)
    end = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f'time: {end-start:.4f}s')
    
    # Store outputs and ground-truth
    np.savetxt('W_true.csv', W_true, delimiter=',')
    np.savetxt('W_est.csv', W_est, delimiter=',')
    np.savetxt('X.csv', X, delimiter=',')

    

    
