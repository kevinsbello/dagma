:py:func:`dagma.utils.simulate_parameter <dagma.utils.simulate_parameter>`
==========================================================================
.. _dagma.utils.simulate_parameter:
.. py:function:: dagma.utils.simulate_parameter(B: numpy.ndarray, w_ranges: List[Tuple[float, float]] = ((-2.0, -0.5), (0.5, 2.0))) -> numpy.ndarray

   Simulate SEM parameters for a DAG.

   :param B: :math:`[d, d]` binary adj matrix of DAG.
   :type B: np.ndarray
   :param w_ranges: disjoint weight ranges, by default :math:`((-2.0, -0.5), (0.5, 2.0))`.
   :type w_ranges: typing.List[typing.Tuple[float,float]], optional

   :returns: :math:`[d, d]` weighted adj matrix of DAG.
   :rtype: np.ndarray



