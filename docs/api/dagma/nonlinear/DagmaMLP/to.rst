to
==
.. py:method:: to(device: Optional[Union[int, Module.to.device]] = ..., dtype: Optional[Union[Module.to.dtype, str]] = ..., non_blocking: bool = ...) -> T
               to(dtype: Union[Module.to.dtype, str], non_blocking: bool = ...) -> T
               to(tensor: torch.Tensor, non_blocking: bool = ...) -> T

   Moves and/or casts the parameters and buffers.

   This can be called as

   .. function:: to(device=None, dtype=None, non_blocking=False)
      :noindex:

   .. function:: to(dtype, non_blocking=False)
      :noindex:

   .. function:: to(tensor, non_blocking=False)
      :noindex:

   .. function:: to(memory_format=torch.channels_last)
      :noindex:

   Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
   floating point or complex :attr:`dtype`\ s. In addition, this method will
   only cast the floating point or complex parameters and buffers to :attr:`dtype`
   (if given). The integral parameters and buffers will be moved
   :attr:`device`, if that is given, but with dtypes unchanged. When
   :attr:`non_blocking` is set, it tries to convert/move asynchronously
   with respect to the host if possible, e.g., moving CPU Tensors with
   pinned memory to CUDA devices.

   See below for examples.

   .. note::
       This method modifies the module in-place.

   :param device: the desired device of the parameters
                  and buffers in this module
   :type device: :class:`torch.device`
   :param dtype: the desired floating point or complex dtype of
                 the parameters and buffers in this module
   :type dtype: :class:`torch.dtype`
   :param tensor: Tensor whose dtype and device are the desired
                  dtype and device for all parameters and buffers in this module
   :type tensor: torch.Tensor
   :param memory_format: the desired memory
                         format for 4D parameters and buffers in this module (keyword
                         only argument)
   :type memory_format: :class:`torch.memory_format`

   :returns: self
   :rtype: Module

   Examples::

       >>> # xdoctest: +IGNORE_WANT("non-deterministic")
       >>> linear = nn.Linear(2, 2)
       >>> linear.weight
       Parameter containing:
       tensor([[ 0.1913, -0.3420],
               [-0.5113, -0.2325]])
       >>> linear.to(torch.double)
       Linear(in_features=2, out_features=2, bias=True)
       >>> linear.weight
       Parameter containing:
       tensor([[ 0.1913, -0.3420],
               [-0.5113, -0.2325]], dtype=torch.float64)
       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
       >>> gpu1 = torch.device("cuda:1")
       >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
       Linear(in_features=2, out_features=2, bias=True)
       >>> linear.weight
       Parameter containing:
       tensor([[ 0.1914, -0.3420],
               [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
       >>> cpu = torch.device("cpu")
       >>> linear.to(cpu)
       Linear(in_features=2, out_features=2, bias=True)
       >>> linear.weight
       Parameter containing:
       tensor([[ 0.1914, -0.3420],
               [-0.5112, -0.2324]], dtype=torch.float16)

       >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
       >>> linear.weight
       Parameter containing:
       tensor([[ 0.3741+0.j,  0.2382+0.j],
               [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
       >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
       tensor([[0.6122+0.j, 0.1150+0.j],
               [0.6122+0.j, 0.1150+0.j],
               [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)


