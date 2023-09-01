:py:function:`dagma.utils.simulate_nonlinear_sem`
============================================
.. py:function:: simulate_nonlinear_sem(B, n, sem_type, noise_scale=None)

   Simulate samples from nonlinear SEM.

   :param B: [d, d] binary adj matrix of DAG
   :type B: np.ndarray
   :param n: num of samples
   :type n: int
   :param sem_type: mlp, mim, gp, gp-add
   :type sem_type: str
   :param noise_scale: scale parameter of additive noise, default all ones
   :type noise_scale: np.ndarray

   :returns: [n, d] sample matrix
   :rtype: X (np.ndarray)

