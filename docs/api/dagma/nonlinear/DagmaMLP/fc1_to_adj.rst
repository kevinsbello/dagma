:py:method:`dagma.nonlinear.DagmaMLP.fc1_to_adj`
=============================================
.. _dagma.nonlinear.DagmaMLP.fc1_to_adj:
.. py:method:: fc1_to_adj() -> numpy.ndarray

   Computes the induced weighted adjacency matrix W from the first FC weights.
   Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

   :returns: :math:`(d,d)` weighted adjacency matrix
   :rtype: np.ndarray

