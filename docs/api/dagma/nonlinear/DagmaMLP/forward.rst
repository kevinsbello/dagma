:py:method:`dagma.nonlinear.DagmaMLP.forward`
==========================================
.. py:method:: forward(x: torch.Tensor) -> torch.Tensor

   Applies the current states of the structural equations to the dataset X

   :param x: Input dataset with shape :math:`(n,d)`.
   :type x: torch.Tensor

   :returns: Result of applying the structural equations to the input data.
             Shape :math:`(n,d)`.
   :rtype: torch.Tensor

