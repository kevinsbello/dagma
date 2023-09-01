:py:method:`dagma.nonlinear.DagmaMLP.type`
=======================================
.. _dagma.nonlinear.DagmaMLP.type:
.. py:method:: type(dst_type: Union[torch.dtype, str]) -> T

   Casts all parameters and buffers to :attr:`dst_type`.

   .. note::
       This method modifies the module in-place.

   :param dst_type: the desired type
   :type dst_type: type or string

   :returns: self
   :rtype: Module

