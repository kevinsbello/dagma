:py:meth:`dagma.locally_connected.LocallyConnected.named_children <dagma.locally_connected.LocallyConnected.named_children>`
============================================================================================================================
.. _dagma.locally_connected.LocallyConnected.named_children:
.. py:method:: dagma.locally_connected.LocallyConnected.named_children() -> Iterator[Tuple[str, Module]]

   Returns an iterator over immediate children modules, yielding both
   the name of the module as well as the module itself.

   :Yields: *(str, Module)* -- Tuple containing a name and child module

   Example::

       >>> # xdoctest: +SKIP("undefined vars")
       >>> for name, module in model.named_children():
       >>>     if name in ['conv4', 'conv5']:
       >>>         print(module)




