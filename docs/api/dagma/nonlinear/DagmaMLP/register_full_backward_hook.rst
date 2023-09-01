:py:method:`dagma.nonlinear.DagmaMLP.register_full_backward_hook`
==============================================================
.. _dagma.nonlinear.DagmaMLP.register_full_backward_hook:
.. py:method:: register_full_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle

   Registers a backward hook on the module.

   The hook will be called every time the gradients with respect to a module
   are computed, i.e. the hook will execute if and only if the gradients with
   respect to module outputs are computed. The hook should have the following
   signature::

       hook(module, grad_input, grad_output) -> tuple(Tensor) or None

   The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
   with respect to the inputs and outputs respectively. The hook should
   not modify its arguments, but it can optionally return a new gradient with
   respect to the input that will be used in place of :attr:`grad_input` in
   subsequent computations. :attr:`grad_input` will only correspond to the inputs given
   as positional arguments and all kwarg arguments are ignored. Entries
   in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
   arguments.

   For technical reasons, when this hook is applied to a Module, its forward function will
   receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
   of each Tensor returned by the Module's forward function.

   .. warning ::
       Modifying inputs or outputs inplace is not allowed when using backward hooks and
       will raise an error.

   :param hook: The user-defined hook to be registered.
   :type hook: Callable
   :param prepend: If true, the provided ``hook`` will be fired before
                   all existing ``backward`` hooks on this
                   :class:`torch.nn.modules.Module`. Otherwise, the provided
                   ``hook`` will be fired after all existing ``backward`` hooks on
                   this :class:`torch.nn.modules.Module`. Note that global
                   ``backward`` hooks registered with
                   :func:`register_module_full_backward_hook` will fire before
                   all hooks registered by this method.
   :type prepend: bool

   :returns:     a handle that can be used to remove the added hook by calling
                 ``handle.remove()``
   :rtype: :class:`torch.utils.hooks.RemovableHandle`

