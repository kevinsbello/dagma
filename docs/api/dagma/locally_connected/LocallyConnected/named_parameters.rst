:py:method:`dagma.locally_connected.LocallyConnected.named_parameters`
===================================================================
.. py:method:: named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]

   Returns an iterator over module parameters, yielding both the
   name of the parameter as well as the parameter itself.

   :param prefix: prefix to prepend to all parameter names.
   :type prefix: str
   :param recurse: if True, then yields parameters of this module
                   and all submodules. Otherwise, yields only parameters that
                   are direct members of this module.
   :type recurse: bool
   :param remove_duplicate: whether to remove the duplicated
                            parameters in the result. Defaults to True.
   :type remove_duplicate: bool, optional

   :Yields: *(str, Parameter)* -- Tuple containing the name and parameter

   Example::

       >>> # xdoctest: +SKIP("undefined vars")
       >>> for name, param in self.named_parameters():
       >>>     if name in ['bias']:
       >>>         print(param.size())


