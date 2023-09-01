:py:method:`dagma.nonlinear.DagmaMLP.get_submodule`
================================================
.. py:method:: get_submodule(target: str) -> Module

   Returns the submodule given by ``target`` if it exists,
   otherwise throws an error.

   For example, let's say you have an ``nn.Module`` ``A`` that
   looks like this:

   .. code-block:: text

       A(
           (net_b): Module(
               (net_c): Module(
                   (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
               )
               (linear): Linear(in_features=100, out_features=200, bias=True)
           )
       )

   (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
   submodule ``net_b``, which itself has two submodules ``net_c``
   and ``linear``. ``net_c`` then has a submodule ``conv``.)

   To check whether or not we have the ``linear`` submodule, we
   would call ``get_submodule("net_b.linear")``. To check whether
   we have the ``conv`` submodule, we would call
   ``get_submodule("net_b.net_c.conv")``.

   The runtime of ``get_submodule`` is bounded by the degree
   of module nesting in ``target``. A query against
   ``named_modules`` achieves the same result, but it is O(N) in
   the number of transitive modules. So, for a simple check to see
   if some submodule exists, ``get_submodule`` should always be
   used.

   :param target: The fully-qualified string name of the submodule
                  to look for. (See above example for how to specify a
                  fully-qualified string.)

   :returns: The submodule referenced by ``target``
   :rtype: torch.nn.Module

   :raises AttributeError: If the target string references an invalid
       path or resolves to something that is not an
       ``nn.Module``

