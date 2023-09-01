:py:method:`dagma.locally_connected.LocallyConnected.named_buffers`
================================================================
.. _dagma.locally_connected.LocallyConnected.named_buffers:
.. py:method:: named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.Tensor]]

   Returns an iterator over module buffers, yielding both the
   name of the buffer as well as the buffer itself.

   :param prefix: prefix to prepend to all buffer names.
   :type prefix: str
   :param recurse: if True, then yields buffers of this module
                   and all submodules. Otherwise, yields only buffers that
                   are direct members of this module. Defaults to True.
   :type recurse: bool, optional
   :param remove_duplicate: whether to remove the duplicated buffers in the result. Defaults to True.
   :type remove_duplicate: bool, optional

   :Yields: *(str, torch.Tensor)* -- Tuple containing the name and buffer

   Example::

       >>> # xdoctest: +SKIP("undefined vars")
       >>> for name, buf in self.named_buffers():
       >>>     if name in ['running_var']:
       >>>         print(buf.size())


