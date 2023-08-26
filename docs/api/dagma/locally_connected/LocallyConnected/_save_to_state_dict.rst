:py:mod:`dagma.locally_connected.LocallyConnected._save_to_state_dict`
======================================================================
.. py:method:: _save_to_state_dict(destination, prefix, keep_vars)

   Saves module state to `destination` dictionary, containing a state
   of the module, but not its descendants. This is called on every
   submodule in :meth:`~torch.nn.Module.state_dict`.

   In rare cases, subclasses can achieve class-specific behavior by
   overriding this method with custom logic.

   :param destination: a dict where state will be stored
   :type destination: dict
   :param prefix: the prefix for parameters and buffers used in this
                  module
   :type prefix: str

