.. py:method:: _h(W: numpy.ndarray, s: float = 1.0) -> Tuple[float, numpy.ndarray]

   Evaluate value and gradient of the logdet acyclicity constraint.

   :param W: :math:`(d,d)` adjacency matrix
   :type W: np.ndarray
   :param s: Controls the domain of M-matrices. Defaults to 1.0.
   :type s: float, optional

   :returns: h value, and gradient of h
   :rtype: typing.Tuple[float, np.ndarray]

