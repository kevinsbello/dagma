:py:func:`dagma.utils.simulate_parameter <dagma.utils.simulate_parameter>`
==========================================================================
.. _dagma.utils.simulate_parameter:
.. py:function:: simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0)))

   Simulate SEM parameters for a DAG.

   :param B: [d, d] binary adj matrix of DAG
   :type B: np.ndarray
   :param w_ranges: disjoint weight ranges
   :type w_ranges: tuple

   :returns: [d, d] weighted adj matrix of DAG
   :rtype: W (np.ndarray)

