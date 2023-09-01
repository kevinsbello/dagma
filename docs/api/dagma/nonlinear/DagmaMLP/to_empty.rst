:py:meth:`dagma.nonlinear.DagmaMLP.to_empty <dagma.nonlinear.DagmaMLP.to_empty>`
================================================================================
.. _dagma.nonlinear.DagmaMLP.to_empty:
.. py:method:: to_empty(*, device: Union[str, Module.to_empty.device]) -> T

   Moves the parameters and buffers to the specified device without copying storage.

   :param device: The desired device of the parameters
                  and buffers in this module.
   :type device: :class:`torch.device`

   :returns: self
   :rtype: Module

