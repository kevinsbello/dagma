:py:meth:`dagma.locally_connected.LocallyConnected.register_load_state_dict_post_hook <dagma.locally_connected.LocallyConnected.register_load_state_dict_post_hook>`
====================================================================================================================================================================
.. _dagma.locally_connected.LocallyConnected.register_load_state_dict_post_hook:
.. py:method:: register_load_state_dict_post_hook(hook)

   Registers a post hook to be run after module's ``load_state_dict``
   is called.

   It should have the following signature::
       hook(module, incompatible_keys) -> None

   The ``module`` argument is the current module that this hook is registered
   on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
   of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
   is a ``list`` of ``str`` containing the missing keys and
   ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

   The given incompatible_keys can be modified inplace if needed.

   Note that the checks performed when calling :func:`load_state_dict` with
   ``strict=True`` are affected by modifications the hook makes to
   ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
   set of keys will result in an error being thrown when ``strict=True``, and
   clearing out both missing and unexpected keys will avoid an error.

   :returns:     a handle that can be used to remove the added hook by calling
                 ``handle.remove()``
   :rtype: :class:`torch.utils.hooks.RemovableHandle`

