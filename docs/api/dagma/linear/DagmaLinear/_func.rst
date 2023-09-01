:py:method:`dagma.linear.DagmaLinear._func`
========================================
.. _dagma.linear.DagmaLinear._func:
.. py:method:: _func(W: numpy.ndarray, mu: float, s: float = 1.0) -> Tuple[float, numpy.ndarray]

   Evaluate value of the penalized objective function.

   :param W: :math:`(d,d)` adjacency matrix
   :type W: np.ndarray
   :param mu: Weight of the score function.
   :type mu: float
   :param s: Controls the domain of M-matrices. Defaults to 1.0.
   :type s: float, optional

   :returns: Objective value, and gradient of the objective
   :rtype: typing.Tuple[float, np.ndarray]

