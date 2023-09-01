:py:method:`dagma.nonlinear.DagmaMLP.xpu`
======================================
.. _dagma.nonlinear.DagmaMLP.xpu:
.. py:method:: xpu(device: Optional[Union[int, Module.xpu.device]] = None) -> T

   Moves all model parameters and buffers to the XPU.

   This also makes associated parameters and buffers different objects. So
   it should be called before constructing optimizer if the module will
   live on XPU while being optimized.

   .. note::
       This method modifies the module in-place.

   :param device: if specified, all parameters will be
                  copied to that device
   :type device: int, optional

   :returns: self
   :rtype: Module

