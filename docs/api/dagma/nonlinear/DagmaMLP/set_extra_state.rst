:py:method:`dagma.nonlinear.DagmaMLP.set_extra_state`
==================================================
.. _dagma.nonlinear.DagmaMLP.set_extra_state:
.. py:method:: set_extra_state(state: Any)

   This function is called from :func:`load_state_dict` to handle any extra state
   found within the `state_dict`. Implement this function and a corresponding
   :func:`get_extra_state` for your module if you need to store extra state within its
   `state_dict`.

   :param state: Extra state from the `state_dict`
   :type state: dict

