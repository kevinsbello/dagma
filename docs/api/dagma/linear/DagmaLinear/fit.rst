:py:mod:`dagma.linear.DagmaLinear.fit`
======================================
.. py:method:: fit(X: numpy.ndarray, lambda1: float, w_threshold: float = 0.3, T: int = 5, mu_init: float = 1.0, mu_factor: float = 0.1, s: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6], warm_iter: int = 30000.0, max_iter: int = 60000.0, lr: float = 0.0003, checkpoint: int = 1000, beta_1: float = 0.99, beta_2: float = 0.999, exclude_edges: Optional[List[Tuple[int, int]]] = None, include_edges: Optional[List[Tuple[int, int]]] = None) -> numpy.ndarray

   Runs the DAGMA algorithm and returns a weighted adjacency matrix.

   :param X: :math:`(n,d)` dataset
   :type X: np.ndarray
   :param lambda1: Coefficient of the L1 penalty
   :type lambda1: float
   :param w_threshold: Removes edges with weight value less than the given threshold. Defaults to 0.3.
   :type w_threshold: float, optional
   :param T: Number of DAGMA iterations. Defaults to 5.
   :type T: int, optional
   :param mu_init: Initial value of :math:`\mu`. Defaults to 1.0.
   :type mu_init: float, optional
   :param mu_factor: Decay factor for :math:`\mu`. Defaults to 0.1.
   :type mu_factor: float, optional
   :param s: Controls the domain of M-matrices. Defaults to [1.0, .9, .8, .7, .6].
   :type s: typing.List[float], optional
   :param warm_iter: Number of iterations for :py:obj:`minimize` for :math:`t < T`. Defaults to 3e4.
   :type warm_iter: int, optional
   :param max_iter: Number of iterations for :py:obj:`minimize` for :math:`t = T`. Defaults to 6e4.
   :type max_iter: int, optional
   :param lr: Learning rate. Defaults to 0.0003.
   :type lr: float, optional
   :param checkpoint: If `verbose` is `True`, then prints to stdout every `checkpoint` iterations. Defaults to 1000.
   :type checkpoint: int, optional
   :param beta_1: Adam hyperparameter. Defaults to 0.99.
   :type beta_1: float, optional
   :param beta_2: Adam hyperparameter. Defaults to 0.999.
   :type beta_2: float, optional
   :param exclude_edges: Tuple of edges that should be excluded from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
   :type exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]], optional
   :param include_edges: Tuple of edges that should be included from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
   :type include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]], optional

   :returns: Estimated DAG from data.
   :rtype: np.ndarray

   .. important::

           If the output of :py:obj:`fit` is not a DAG, then the user should try larger value of `T` before raising an issue.

   .. warning::

           While DAGMA ensures to exclude such edges in `exclude_edges`, the method does not guarantees that all edges
           will be included from `included edges`.

