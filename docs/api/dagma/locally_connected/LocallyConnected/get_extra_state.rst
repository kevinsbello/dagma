:py:meth:`dagma.locally_connected.LocallyConnected.get_extra_state <dagma.locally_connected.LocallyConnected.get_extra_state>`
==============================================================================================================================
.. _dagma.locally_connected.LocallyConnected.get_extra_state:
.. py:method:: get_extra_state() -> Any

   Returns any extra state to include in the module's state_dict.
   Implement this and a corresponding :func:`set_extra_state` for your module
   if you need to store extra state. This function is called when building the
   module's `state_dict()`.

   Note that extra state should be picklable to ensure working serialization
   of the state_dict. We only provide provide backwards compatibility guarantees
   for serializing Tensors; other objects may break backwards compatibility if
   their serialized pickled form changes.

   :returns: Any extra state to store in the module's state_dict
   :rtype: object

