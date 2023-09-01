:py:meth:`dagma.nonlinear.DagmaMLP.eval <dagma.nonlinear.DagmaMLP.eval>`
========================================================================
.. _dagma.nonlinear.DagmaMLP.eval:
.. py:method:: dagma.nonlinear.DagmaMLP.eval() -> T

   Sets the module in evaluation mode.

   This has any effect only on certain modules. See documentations of
   particular modules for details of their behaviors in training/evaluation
   mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
   etc.

   This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

   See :ref:`locally-disable-grad-doc` for a comparison between
   `.eval()` and several similar mechanisms that may be confused with it.

   :returns: self
   :rtype: Module



