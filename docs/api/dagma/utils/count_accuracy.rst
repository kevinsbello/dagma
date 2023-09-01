:py:func:`dagma.utils.count_accuracy <dagma.utils.count_accuracy>`
==================================================================
.. _dagma.utils.count_accuracy:
.. py:function:: dagma.utils.count_accuracy(B_true: numpy.ndarray, B_est: numpy.ndarray) -> dict

   Compute various accuracy metrics for B_est.

   | true positive = predicted association exists in condition in correct direction
   | reverse = predicted association exists in condition in opposite direction
   | false positive = predicted association does not exist in condition

   :param B_true: :math:`[d, d]` ground truth graph, :math:`\{0, 1\}`.
   :type B_true: np.ndarray
   :param B_est: :math:`[d, d]` estimate, :math:`\{0, 1, -1\}`, -1 is undirected edge in CPDAG.
   :type B_est: np.ndarray

   :returns: | fdr: (reverse + false positive) / prediction positive
             | tpr: (true positive) / condition positive
             | fpr: (reverse + false positive) / condition negative
             | shd: undirected extra + undirected missing + reverse
             | nnz: prediction positive
   :rtype: dict



