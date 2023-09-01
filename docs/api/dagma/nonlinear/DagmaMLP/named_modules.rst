:py:meth:`dagma.nonlinear.DagmaMLP.named_modules <dagma.nonlinear.DagmaMLP.named_modules>`
==========================================================================================
.. _dagma.nonlinear.DagmaMLP.named_modules:
.. py:method:: dagma.nonlinear.DagmaMLP.named_modules(memo: Optional[Set[Module]] = None, prefix: str = '', remove_duplicate: bool = True)

   Returns an iterator over all modules in the network, yielding
   both the name of the module as well as the module itself.

   :param memo: a memo to store the set of modules already added to the result
   :param prefix: a prefix that will be added to the name of the module
   :param remove_duplicate: whether to remove the duplicated module instances in the result
                            or not

   :Yields: *(str, Module)* -- Tuple of name and module

   .. note::

      Duplicate modules are returned only once. In the following
      example, ``l`` will be returned only once.

   Example::

       >>> l = nn.Linear(2, 2)
       >>> net = nn.Sequential(l, l)
       >>> for idx, m in enumerate(net.named_modules()):
       ...     print(idx, '->', m)

       0 -> ('', Sequential(
         (0): Linear(in_features=2, out_features=2, bias=True)
         (1): Linear(in_features=2, out_features=2, bias=True)
       ))
       1 -> ('0', Linear(in_features=2, out_features=2, bias=True))




