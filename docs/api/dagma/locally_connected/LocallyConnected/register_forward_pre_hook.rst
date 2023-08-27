register_forward_pre_hook
=========================
.. py:method:: register_forward_pre_hook(hook: Union[Callable[[T, Tuple[Any, Ellipsis]], Optional[Any]], Callable[[T, Tuple[Any, Ellipsis], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle

   Registers a forward pre-hook on the module.

   The hook will be called every time before :func:`forward` is invoked.


   If ``with_kwargs`` is false or not specified, the input contains only
   the positional arguments given to the module. Keyword arguments won't be
   passed to the hooks and only to the ``forward``. The hook can modify the
   input. User can either return a tuple or a single modified value in the
   hook. We will wrap the value into a tuple if a single value is returned
   (unless that value is already a tuple). The hook should have the
   following signature::

       hook(module, args) -> None or modified input

   If ``with_kwargs`` is true, the forward pre-hook will be passed the
   kwargs given to the forward function. And if the hook modifies the
   input, both the args and kwargs should be returned. The hook should have
   the following signature::

       hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

   :param hook: The user defined hook to be registered.
   :type hook: Callable
   :param prepend: If true, the provided ``hook`` will be fired before
                   all existing ``forward_pre`` hooks on this
                   :class:`torch.nn.modules.Module`. Otherwise, the provided
                   ``hook`` will be fired after all existing ``forward_pre`` hooks
                   on this :class:`torch.nn.modules.Module`. Note that global
                   ``forward_pre`` hooks registered with
                   :func:`register_module_forward_pre_hook` will fire before all
                   hooks registered by this method.
                   Default: ``False``
   :type prepend: bool
   :param with_kwargs: If true, the ``hook`` will be passed the kwargs
                       given to the forward function.
                       Default: ``False``
   :type with_kwargs: bool

   :returns:     a handle that can be used to remove the added hook by calling
                 ``handle.remove()``
   :rtype: :class:`torch.utils.hooks.RemovableHandle`

