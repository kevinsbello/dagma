:py:method:`dagma.locally_connected.LocallyConnected.zero_grad`
============================================================
.. _dagma.locally_connected.LocallyConnected.zero_grad:
.. py:method:: zero_grad(set_to_none: bool = True) -> None

   Sets gradients of all model parameters to zero. See similar function
   under :class:`torch.optim.Optimizer` for more context.

   :param set_to_none: instead of setting to zero, set the grads to None.
                       See :meth:`torch.optim.Optimizer.zero_grad` for details.
   :type set_to_none: bool

