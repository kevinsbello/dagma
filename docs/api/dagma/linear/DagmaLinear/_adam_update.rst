_adam_update
============
.. py:method:: _adam_update(grad: numpy.ndarray, iter: int, beta_1: float, beta_2: float) -> numpy.ndarray

   Performs one update of Adam.

   :param grad: Current gradient of the objective.
   :type grad: np.ndarray
   :param iter: Current iteration number.
   :type iter: int
   :param beta_1: Adam hyperparameter.
   :type beta_1: float
   :param beta_2: Adam hyperparameter.
   :type beta_2: float

   :returns: Updates the gradient by the Adam method.
   :rtype: np.ndarray

