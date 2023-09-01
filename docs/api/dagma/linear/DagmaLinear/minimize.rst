:py:method:`dagma.linear.DagmaLinear.minimize`
===========================================
.. py:method:: minimize(W: numpy.ndarray, mu: float, max_iter: int, s: float, lr: float, tol: float = 1e-06, beta_1: float = 0.99, beta_2: float = 0.999, pbar: tqdm.auto.tqdm = tqdm()) -> Tuple[numpy.ndarray, bool]

   Solves the optimization problem:
       .. math::
           \arg\min_{W \in \mathbb{W}^s} \mu \cdot Q(W; \mathbf{X}) + h(W),
   where :math:`Q` is the score function. This problem is solved via (sub)gradient descent, where the initial
   point is `W`.

   :param W: Initial point of (sub)gradient descent.
   :type W: np.ndarray
   :param mu: Weights the score function.
   :type mu: float
   :param max_iter: Maximum number of (sub)gradient iterations.
   :type max_iter: int
   :param s: Number that controls the domain of M-matrices.
   :type s: float
   :param lr: Learning rate.
   :type lr: float
   :param tol: Tolerance to admit convergence. Defaults to 1e-6.
   :type tol: float, optional
   :param beta_1: Hyperparamter for Adam. Defaults to 0.99.
   :type beta_1: float, optional
   :param beta_2: Hyperparamter for Adam. Defaults to 0.999.
   :type beta_2: float, optional
   :param pbar: Controls bar progress. Defaults to ``tqdm()``.
   :type pbar: tqdm, optional

   :returns: Returns an adjacency matrix until convergence or `max_iter` is reached.
             A boolean flag is returned to point success of the optimization. This can be False when at any iteration, the current
             W point went outside of the domain of M-matrices.
   :rtype: typing.Tuple[np.ndarray, bool]

