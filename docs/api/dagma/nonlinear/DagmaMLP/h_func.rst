:py:meth:`dagma.nonlinear.DagmaMLP.h_func <dagma.nonlinear.DagmaMLP.h_func>`
============================================================================
.. _dagma.nonlinear.DagmaMLP.h_func:
.. py:method:: h_func(s: float = 1.0) -> torch.Tensor

   Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

   :param s: Controls the domain of M-matrices, by default 1.0
   :type s: float, optional

   :returns: A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
   :rtype: torch.Tensor

