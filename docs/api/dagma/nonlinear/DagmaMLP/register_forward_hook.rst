:py:meth:`dagma.nonlinear.DagmaMLP.register_forward_hook <dagma.nonlinear.DagmaMLP.register_forward_hook>`
==========================================================================================================
.. _dagma.nonlinear.DagmaMLP.register_forward_hook:
.. py:method:: dagma.nonlinear.DagmaMLP.register_forward_hook(hook: Union[Callable[[T, Tuple[Any, Ellipsis], Any], Optional[Any]], Callable[[T, Tuple[Any, Ellipsis], Dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle

   Registers a forward hook on the module.

   The hook will be called every time after :func:`forward` has computed an output.

   If ``with_kwargs`` is ``False`` or not specified, the input contains only
   the positional arguments given to the module. Keyword arguments won't be
   passed to the hooks and only to the ``forward``. The hook can modify the
   output. It can modify the input inplace but it will not have effect on
   forward since this is called after :func:`forward` is called. The hook
   should have the following signature::

       hook(module, args, output) -> None or modified output

   If ``with_kwargs`` is ``True``, the forward hook will be passed the
   ``kwargs`` given to the forward function and be expected to return the
   output possibly modified. The hook should have the following signature::

       hook(module, args, kwargs, output) -> None or modified output

   :param hook: The user defined hook to be registered.
   :type hook: Callable
   :param prepend: If ``True``, the provided ``hook`` will be fired
                   before all existing ``forward`` hooks on this
                   :class:`torch.nn.modules.Module`. Otherwise, the provided
                   ``hook`` will be fired after all existing ``forward`` hooks on
                   this :class:`torch.nn.modules.Module`. Note that global
                   ``forward`` hooks registered with
                   :func:`register_module_forward_hook` will fire before all hooks
                   registered by this method.
                   Default: ``False``
   :type prepend: bool
   :param with_kwargs: If ``True``, the ``hook`` will be passed the
                       kwargs given to the forward function.
                       Default: ``False``
   :type with_kwargs: bool

   :returns:     a handle that can be used to remove the added hook by calling
                 ``handle.remove()``
   :rtype: :class:`torch.utils.hooks.RemovableHandle`



