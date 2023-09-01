:py:func:`dagma.utils.simulate_dag <dagma.utils.simulate_dag>`
==============================================================
.. _dagma.utils.simulate_dag:
.. py:function:: dagma.utils.simulate_dag(d: int, s0: int, graph_type: str) -> numpy.ndarray

   Simulate random DAG with some expected number of edges.

   :param d: num of nodes
   :type d: int
   :param s0: expected num of edges
   :type s0: int
   :param graph_type: One of ``["ER", "SF", "BP"]``
   :type graph_type: str

   :returns: :math:`(d, d)` binary adj matrix of DAG
   :rtype: numpy.ndarray



