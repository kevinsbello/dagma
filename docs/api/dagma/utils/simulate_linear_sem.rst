simulate_linear_sem
===================
.. py:function:: simulate_linear_sem(W, n, sem_type, noise_scale=None)

   Simulate samples from linear SEM with specified type of noise.

   For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

   :param W: [d, d] weighted adj matrix of DAG
   :type W: np.ndarray
   :param n: num of samples, n=inf mimics population risk
   :type n: int
   :param sem_type: gauss, exp, gumbel, uniform, logistic, poisson
   :type sem_type: str
   :param noise_scale: scale parameter of additive noise, default all ones
   :type noise_scale: np.ndarray

   :returns: [n, d] sample matrix, [d, d] if n=inf
   :rtype: X (np.ndarray)

