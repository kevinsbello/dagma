:py:method:`dagma.nonlinear.DagmaNonlinear.log_mse_loss`
=====================================================
.. _dagma.nonlinear.DagmaNonlinear.log_mse_loss:
.. py:method:: log_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor

   Computes the logarithm of the MSE loss:
       .. math::
           \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)

   :param output: :math:`(n,d)` output of the model
   :type output: torch.Tensor
   :param target: :math:`(n,d)` input dataset
   :type target: torch.Tensor

   :returns: A scalar value of the loss.
   :rtype: torch.Tensor

