:py:method:`dagma.locally_connected.LocallyConnected.state_dict`
=============================================================
.. _dagma.locally_connected.LocallyConnected.state_dict:
.. py:method:: state_dict(*, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination
               state_dict(*, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]

   Returns a dictionary containing references to the whole state of the module.

   Both parameters and persistent buffers (e.g. running averages) are
   included. Keys are corresponding parameter and buffer names.
   Parameters and buffers set to ``None`` are not included.

   .. note::
       The returned object is a shallow copy. It contains references
       to the module's parameters and buffers.

   .. warning::
       Currently ``state_dict()`` also accepts positional arguments for
       ``destination``, ``prefix`` and ``keep_vars`` in order. However,
       this is being deprecated and keyword arguments will be enforced in
       future releases.

   .. warning::
       Please avoid the use of argument ``destination`` as it is not
       designed for end-users.

   :param destination: If provided, the state of module will
                       be updated into the dict and the same object is returned.
                       Otherwise, an ``OrderedDict`` will be created and returned.
                       Default: ``None``.
   :type destination: dict, optional
   :param prefix: a prefix added to parameter and buffer
                  names to compose the keys in state_dict. Default: ``''``.
   :type prefix: str, optional
   :param keep_vars: by default the :class:`~torch.Tensor` s
                     returned in the state dict are detached from autograd. If it's
                     set to ``True``, detaching will not be performed.
                     Default: ``False``.
   :type keep_vars: bool, optional

   :returns:     a dictionary containing a whole state of the module
   :rtype: dict

   Example::

       >>> # xdoctest: +SKIP("undefined vars")
       >>> module.state_dict().keys()
       ['bias', 'weight']


