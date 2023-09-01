:py:class:`dagma.linear.DagmaLinear <dagma.linear.DagmaLinear>`
===============================================================

.. _dagma.linear.DagmaLinear:

.. py:class:: dagma.linear.DagmaLinear(loss_type: str, verbose: bool = False, dtype: type = np.float64)


   A Python object that contains the implementation of DAGMA for linear models using numpy and scipy.

   :param loss_type: One of ["l2", "logistic"]. ``l2`` refers to the least squares loss, while ``logistic``
                     refers to the logistic loss. For continuous data: use ``l2``. For discrete 0/1 data: use ``logistic``.
   :type loss_type: str
   :param verbose: If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
                   as defined in :py:meth:`~dagma.linear.DagmaLinear.fit`. Defaults to ``False``.
   :type verbose: bool, optional
   :param dtype: Defines the float precision, for large number of nodes it is recommened to use ``np.float64``.
                 Defaults to ``np.float64``.
   :type dtype: type, optional

   Methods
   ~~~~~~~

   .. autoapisummary::

      dagma.linear.DagmaLinear._score
      dagma.linear.DagmaLinear._h
      dagma.linear.DagmaLinear._func
      dagma.linear.DagmaLinear._adam_update
      dagma.linear.DagmaLinear.minimize
      dagma.linear.DagmaLinear.fit

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :hidden:

   _score<_score>
   _h<_h>
   _func<_func>
   _adam_update<_adam_update>
   minimize<minimize>
   fit<fit>

