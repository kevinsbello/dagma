.. py:method:: train(mode: bool = True) -> T

   Sets the module in training mode.

   This has any effect only on certain modules. See documentations of
   particular modules for details of their behaviors in training/evaluation
   mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
   etc.

   :param mode: whether to set training mode (``True``) or evaluation
                mode (``False``). Default: ``True``.
   :type mode: bool

   :returns: self
   :rtype: Module

