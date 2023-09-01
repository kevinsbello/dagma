:py:method:`dagma.nonlinear.DagmaMLP.register_state_dict_pre_hook`
===============================================================
.. _dagma.nonlinear.DagmaMLP.register_state_dict_pre_hook:
.. py:method:: register_state_dict_pre_hook(hook)

   These hooks will be called with arguments: ``self``, ``prefix``,
   and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered
   hooks can be used to perform pre-processing before the ``state_dict``
   call is made.

