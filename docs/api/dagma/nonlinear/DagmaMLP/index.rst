:py:class:`dagma.nonlinear.DagmaMLP`
==================================

.. _dagma.nonlinear.DagmaMLP:

.. py:class:: DagmaMLP(dims: List[int], bias: bool = True, dtype: torch.dtype = torch.double)


   Bases: :py:obj:`torch.nn.Module`

   Class that models the structural equations for the causal graph using MLPs.

   :param dims: Number of neurons in hidden layers of each MLP representing each structural equation.
   :type dims: typing.List[int]
   :param bias: Flag whether to consider bias or not, by default ``True``
   :type bias: bool, optional
   :param dtype: Float precision, by default ``torch.double``
   :type dtype: torch.dtype, optional

   Methods
   ~~~~~~~

   .. autoapisummary::

      dagma.nonlinear.DagmaMLP.forward
      dagma.nonlinear.DagmaMLP.h_func
      dagma.nonlinear.DagmaMLP.fc1_l1_reg
      dagma.nonlinear.DagmaMLP.fc1_to_adj

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :hidden:

   forward<forward>
   h_func<h_func>
   fc1_l1_reg<fc1_l1_reg>
   fc1_to_adj<fc1_to_adj>
