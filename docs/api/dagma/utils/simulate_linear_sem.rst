:py:func:`dagma.utils.simulate_linear_sem <dagma.utils.simulate_linear_sem>`
============================================================================
.. _dagma.utils.simulate_linear_sem:
.. py:function:: dagma.utils.simulate_linear_sem(W: numpy.ndarray, n: int, sem_type: str, noise_scale: Optional[Union[float, List[float]]] = None) -> numpy.ndarray

   Simulate samples from linear SEM with specified type of noise.
   For ``uniform``, noise :math:`z \sim \mathrm{uniform}(-a, a)`, where :math:`a` is the ``noise_scale``.

   :param W: :math:`[d, d]` weighted adj matrix of DAG.
   :type W: np.ndarray
   :param n: num of samples. When ``n=inf`` mimics the population risk, only for Gaussian noise.
   :type n: int
   :param sem_type: ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
   :type sem_type: str
   :param noise_scale: scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.
   :type noise_scale: typing.Optional[typing.Union[float,typing.List[float]]], optional

   :returns: :math:`[n, d]` sample matrix, :math:`[d, d]` if ``n=inf``.
   :rtype: np.ndarray



