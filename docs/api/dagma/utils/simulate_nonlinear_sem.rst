:py:func:`dagma.utils.simulate_nonlinear_sem <dagma.utils.simulate_nonlinear_sem>`
==================================================================================
.. _dagma.utils.simulate_nonlinear_sem:
.. py:function:: dagma.utils.simulate_nonlinear_sem(B: numpy.ndarray, n: int, sem_type: str, noise_scale: Optional[Union[float, List[float]]] = None) -> numpy.ndarray

   Simulate samples from nonlinear SEM.

   :param B: :math:`[d, d]` binary adj matrix of DAG.
   :type B: np.ndarray
   :param n: num of samples
   :type n: int
   :param sem_type: ``mlp``, ``mim``, ``gp``, ``gp-add``
   :type sem_type: str
   :param noise_scale: scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.
   :type noise_scale: typing.Optional[typing.Union[float,typing.List[float]]], optional

   :returns: :math:`[n, d]` sample matrix.
   :rtype: np.ndarray



