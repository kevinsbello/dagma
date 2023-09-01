:py:meth:`dagma.locally_connected.LocallyConnected.add_module <dagma.locally_connected.LocallyConnected.add_module>`
====================================================================================================================
.. _dagma.locally_connected.LocallyConnected.add_module:
.. py:method:: add_module(name: str, module: Optional[Module]) -> None

   Adds a child module to the current module.

   The module can be accessed as an attribute using the given name.

   :param name: name of the child module. The child module can be
                accessed from this module using the given name
   :type name: str
   :param module: child module to be added to the module.
   :type module: Module

