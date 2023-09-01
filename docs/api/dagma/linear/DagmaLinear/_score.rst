:py:meth:`dagma.linear.DagmaLinear._score <dagma.linear.DagmaLinear._score>`
============================================================================
.. _dagma.linear.DagmaLinear._score:
.. py:method:: _score(W: numpy.ndarray) -> Tuple[float, numpy.ndarray]

   Evaluate value and gradient of the score function.

   :param W: :math:`(d,d)` adjacency matrix
   :type W: np.ndarray

   :returns: loss value, and gradient of the loss function
   :rtype: typing.Tuple[float, np.ndarray]

