:py:meth:`dagma.nonlinear.DagmaNonlinear.fit <dagma.nonlinear.DagmaNonlinear.fit>`
==================================================================================
.. _dagma.nonlinear.DagmaNonlinear.fit:
.. py:method:: dagma.nonlinear.DagmaNonlinear.fit(X: Union[torch.Tensor, numpy.ndarray], lambda1: float = 0.02, lambda2: float = 0.005, T: int = 4, mu_init: float = 0.1, mu_factor: float = 0.1, s: float = 1.0, warm_iter: int = 50000.0, max_iter: int = 80000.0, lr: float = 0.0002, w_threshold: float = 0.3, checkpoint: int = 1000) -> numpy.ndarray

   Runs the DAGMA algorithm and fits the model to the dataset.

   :param X: :math:`(n,d)` dataset.
   :type X: typing.Union[torch.Tensor, np.ndarray]
   :param lambda1: Coefficient of the L1 penalty, by default .02.
   :type lambda1: float, optional
   :param lambda2: Coefficient of the L2 penalty, by default .005.
   :type lambda2: float, optional
   :param T: Number of DAGMA iterations, by default 4.
   :type T: int, optional
   :param mu_init: Initial value of :math:`\mu`, by default 0.1.
   :type mu_init: float, optional
   :param mu_factor: Decay factor for :math:`\mu`, by default .1.
   :type mu_factor: float, optional
   :param s: Controls the domain of M-matrices, by default 1.0.
   :type s: float, optional
   :param warm_iter: Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t < T`, by default 5e4.
   :type warm_iter: int, optional
   :param max_iter: Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t = T`, by default 8e4.
   :type max_iter: int, optional
   :param lr: Learning rate, by default .0002.
   :type lr: float, optional
   :param w_threshold: Removes edges with weight value less than the given threshold, by default 0.3.
   :type w_threshold: float, optional
   :param checkpoint: If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.
   :type checkpoint: int, optional

   :returns: Estimated DAG from data.
   :rtype: np.ndarray

   .. important::

       If the output of :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8)
       before raising an issue in github.



