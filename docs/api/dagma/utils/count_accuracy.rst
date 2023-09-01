:py:function:`dagma.utils.count_accuracy`
====================================
.. py:function:: count_accuracy(B_true, B_est)

   Compute various accuracy metrics for B_est.

   true positive = predicted association exists in condition in correct direction
   reverse = predicted association exists in condition in opposite direction
   false positive = predicted association does not exist in condition

   :param B_true: [d, d] ground truth graph, {0, 1}
   :type B_true: np.ndarray
   :param B_est: [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
   :type B_est: np.ndarray

   :returns: (reverse + false positive) / prediction positive
             tpr: (true positive) / condition positive
             fpr: (reverse + false positive) / condition negative
             shd: undirected extra + undirected missing + reverse
             nnz: prediction positive
   :rtype: fdr

