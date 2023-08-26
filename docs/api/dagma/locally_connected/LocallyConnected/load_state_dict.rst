:py:mod:`dagma.locally_connected.LocallyConnected.load_state_dict`
==================================================================
.. py:method:: load_state_dict(state_dict: Mapping[str, Any], strict: bool = True)

   Copies parameters and buffers from :attr:`state_dict` into
   this module and its descendants. If :attr:`strict` is ``True``, then
   the keys of :attr:`state_dict` must exactly match the keys returned
   by this module's :meth:`~torch.nn.Module.state_dict` function.

   :param state_dict: a dict containing parameters and
                      persistent buffers.
   :type state_dict: dict
   :param strict: whether to strictly enforce that the keys
                  in :attr:`state_dict` match the keys returned by this module's
                  :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
   :type strict: bool, optional

   :returns:     * **missing_keys** is a list of str containing the missing keys
                 * **unexpected_keys** is a list of str containing the unexpected keys
   :rtype: ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields

   .. note::

      If a parameter or buffer is registered as ``None`` and its corresponding key
      exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
      ``RuntimeError``.

