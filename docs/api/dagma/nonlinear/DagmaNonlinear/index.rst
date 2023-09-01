:py:class:`dagma.nonlinear.DagmaNonlinear`
========================================

.. py:class:: DagmaNonlinear(model: torch.nn.Module, verbose: bool = False, dtype: torch.dtype = torch.double)


   Class that implements the DAGMA algorithm

   :param model: Neural net that models the structural equations.
   :type model: nn.Module
   :param verbose: If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
                   as defined in :py:method:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
   :type verbose: bool, optional
   :param dtype: float number precision, by default ``torch.double``.
   :type dtype: torch.dtype, optional

   Methods
   ~~~~~~~

   .. autoapisummary::

      dagma.nonlinear.DagmaNonlinear.log_mse_loss
      dagma.nonlinear.DagmaNonlinear.minimize
      dagma.nonlinear.DagmaNonlinear.fit

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :hidden:

   log_mse_loss<log_mse_loss>
   minimize<minimize>
   fit<fit>
