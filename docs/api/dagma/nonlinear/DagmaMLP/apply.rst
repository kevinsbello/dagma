:py:method:`dagma.nonlinear.DagmaMLP.apply`
========================================
.. _dagma.nonlinear.DagmaMLP.apply:
.. py:method:: apply(fn: Callable[[Module], None]) -> T

   Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
   as well as self. Typical use includes initializing the parameters of a model
   (see also :ref:`nn-init-doc`).

   :param fn: function to be applied to each submodule
   :type fn: :class:`Module` -> None

   :returns: self
   :rtype: Module

   Example::

       >>> @torch.no_grad()
       >>> def init_weights(m):
       >>>     print(m)
       >>>     if type(m) == nn.Linear:
       >>>         m.weight.fill_(1.0)
       >>>         print(m.weight)
       >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
       >>> net.apply(init_weights)
       Linear(in_features=2, out_features=2, bias=True)
       Parameter containing:
       tensor([[1., 1.],
               [1., 1.]], requires_grad=True)
       Linear(in_features=2, out_features=2, bias=True)
       Parameter containing:
       tensor([[1., 1.],
               [1., 1.]], requires_grad=True)
       Sequential(
         (0): Linear(in_features=2, out_features=2, bias=True)
         (1): Linear(in_features=2, out_features=2, bias=True)
       )


