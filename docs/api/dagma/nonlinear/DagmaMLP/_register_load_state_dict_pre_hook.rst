:py:method:`dagma.nonlinear.DagmaMLP._register_load_state_dict_pre_hook`
=====================================================================
.. _dagma.nonlinear.DagmaMLP._register_load_state_dict_pre_hook:
.. py:method:: _register_load_state_dict_pre_hook(hook, with_module=False)

   These hooks will be called with arguments: `state_dict`, `prefix`,
   `local_metadata`, `strict`, `missing_keys`, `unexpected_keys`,
   `error_msgs`, before loading `state_dict` into `self`. These arguments
   are exactly the same as those of `_load_from_state_dict`.

   If ``with_module`` is ``True``, then the first argument to the hook is
   an instance of the module.

   :param hook: Callable hook that will be invoked before
                loading the state dict.
   :type hook: Callable
   :param with_module: Whether or not to pass the module
                       instance to the hook as the first parameter.
   :type with_module: bool, optional

