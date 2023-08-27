:py:mod:`dagma.nonlinear.DagmaMLP.register_full_backward_pre_hook`
==================================================================
.. py:method:: register_full_backward_pre_hook(hook: Callable[[Module, _grad_t], Union[None, _grad_t]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle

   Registers a backward pre-hook on the module.

   The hook will be called every time the gradients for the module are computed.
   The hook should have the following signature::

       hook(module, grad_output) -> Tensor or None

   The :attr:`grad_output` is a tuple. The hook should
   not modify its arguments, but it can optionally return a new gradient with
   respect to the output that will be used in place of :attr:`grad_output` in
   subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
   all non-Tensor arguments.

   For technical reasons, when this hook is applied to a Module, its forward function will
   receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
   of each Tensor returned by the Module's forward function.

   .. warning ::
       Modifying inputs inplace is not allowed when using backward hooks and
       will raise an error.

   :param hook: The user-defined hook to be registered.
   :type hook: Callable
   :param prepend: If true, the provided ``hook`` will be fired before
                   all existing ``backward_pre`` hooks on this
                   :class:`torch.nn.modules.Module`. Otherwise, the provided
                   ``hook`` will be fired after all existing ``backward_pre`` hooks
                   on this :class:`torch.nn.modules.Module`. Note that global
                   ``backward_pre`` hooks registered with
                   :func:`register_module_full_backward_pre_hook` will fire before
                   all hooks registered by this method.
   :type prepend: bool

   :returns:     a handle that can be used to remove the added hook by calling
                 ``handle.remove()``
   :rtype: :class:`torch.utils.hooks.RemovableHandle`

