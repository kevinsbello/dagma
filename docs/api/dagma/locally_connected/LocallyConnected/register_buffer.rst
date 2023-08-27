register_buffer
===============
.. py:method:: register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None

   Adds a buffer to the module.

   This is typically used to register a buffer that should not to be
   considered a model parameter. For example, BatchNorm's ``running_mean``
   is not a parameter, but is part of the module's state. Buffers, by
   default, are persistent and will be saved alongside parameters. This
   behavior can be changed by setting :attr:`persistent` to ``False``. The
   only difference between a persistent buffer and a non-persistent buffer
   is that the latter will not be a part of this module's
   :attr:`state_dict`.

   Buffers can be accessed as attributes using given names.

   :param name: name of the buffer. The buffer can be accessed
                from this module using the given name
   :type name: str
   :param tensor: buffer to be registered. If ``None``, then operations
                  that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                  the buffer is **not** included in the module's :attr:`state_dict`.
   :type tensor: Tensor or None
   :param persistent: whether the buffer is part of this module's
                      :attr:`state_dict`.
   :type persistent: bool

   Example::

       >>> # xdoctest: +SKIP("undefined vars")
       >>> self.register_buffer('running_mean', torch.zeros(num_features))


