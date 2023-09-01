:py:meth:`dagma.locally_connected.LocallyConnected.type <dagma.locally_connected.LocallyConnected.type>`
========================================================================================================
.. _dagma.locally_connected.LocallyConnected.type:
.. py:method:: dagma.locally_connected.LocallyConnected.type(dst_type: Union[torch.dtype, str]) -> T

   Casts all parameters and buffers to :attr:`dst_type`.

   .. note::
       This method modifies the module in-place.

   :param dst_type: the desired type
   :type dst_type: type or string

   :returns: self
   :rtype: Module



