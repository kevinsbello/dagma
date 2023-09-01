:py:method:`dagma.nonlinear.DagmaMLP.parameters`
=============================================
.. py:method:: parameters(recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]

   Returns an iterator over module parameters.

   This is typically passed to an optimizer.

   :param recurse: if True, then yields parameters of this module
                   and all submodules. Otherwise, yields only parameters that
                   are direct members of this module.
   :type recurse: bool

   :Yields: *Parameter* -- module parameter

   Example::

       >>> # xdoctest: +SKIP("undefined vars")
       >>> for param in model.parameters():
       >>>     print(type(param), param.size())
       <class 'torch.Tensor'> (20L,)
       <class 'torch.Tensor'> (20L, 1L, 5L, 5L)


