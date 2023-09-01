:py:meth:`dagma.nonlinear.DagmaNonlinear.minimize <dagma.nonlinear.DagmaNonlinear.minimize>`
============================================================================================
.. _dagma.nonlinear.DagmaNonlinear.minimize:
.. py:method:: dagma.nonlinear.DagmaNonlinear.minimize(max_iter: float, lr: float, lambda1: float, lambda2: float, mu: float, s: float, lr_decay: float = False, tol: float = 1e-06, pbar: tqdm.auto.tqdm = tqdm()) -> bool

   Solves the optimization problem:
       .. math::
           \arg\min_{\Theta \in \mathbb{W}(\Theta)^s} \mu \cdot Q(\Theta; \mathbf{X}) + h(W(\Theta)),
   where :math:`Q` is the score function, and :math:`W(\Theta)` is the induced weighted adjacency matrix
   from the model parameters.
   This problem is solved via (sub)gradient descent using adam acceleration.

   :param max_iter: Maximum number of (sub)gradient iterations.
   :type max_iter: float
   :param lr: Learning rate.
   :type lr: float
   :param lambda1: L1 penalty coefficient. Only applies to the parameters that induce the weighted adjacency matrix.
   :type lambda1: float
   :param lambda2: L2 penalty coefficient. Applies to all the model parameters.
   :type lambda2: float
   :param mu: Weights the score function.
   :type mu: float
   :param s: Controls the domain of M-matrices.
   :type s: float
   :param lr_decay: If ``True``, an exponential decay scheduling is used. By default ``False``.
   :type lr_decay: float, optional
   :param tol: Tolerance to admit convergence. Defaults to 1e-6.
   :type tol: float, optional
   :param pbar: Controls bar progress. Defaults to ``tqdm()``.
   :type pbar: tqdm, optional

   :returns: ``True`` if the optimization succeded. This can be ``False`` when at any iteration, the model's adjacency matrix
             got outside of the domain of M-matrices.
   :rtype: bool



