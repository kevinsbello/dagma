:py:meth:`dagma.locally_connected.LocallyConnected.register_backward_hook <dagma.locally_connected.LocallyConnected.register_backward_hook>`
============================================================================================================================================
.. _dagma.locally_connected.LocallyConnected.register_backward_hook:
.. py:method:: dagma.locally_connected.LocallyConnected.register_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]]) -> torch.utils.hooks.RemovableHandle

   Registers a backward hook on the module.

   This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
   the behavior of this function will change in future versions.

   :returns:     a handle that can be used to remove the added hook by calling
                 ``handle.remove()``
   :rtype: :class:`torch.utils.hooks.RemovableHandle`



