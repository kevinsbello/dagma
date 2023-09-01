:py:class:`dagma.locally_connected.LocallyConnected <dagma.locally_connected.LocallyConnected>`
===============================================================================================

.. _dagma.locally_connected.LocallyConnected:

.. py:class:: dagma.locally_connected.LocallyConnected(num_linear: int, input_features: int, output_features: int, bias: bool = True)


   Bases: :py:obj:`torch.nn.Module`

   Implements a local linear layer, i.e. Conv1dLocal() with filter size 1.

   :param num_linear: num of local linear layers, i.e.
   :type num_linear: int
   :param input_features: m1
   :type input_features: int
   :param output_features: m2
   :type output_features: int
   :param bias: Whether to include bias or not. Default: ``True``.
   :type bias: bool, optional

   .. attribute:: weight



      :type: [d, m1, m2]

   .. attribute:: bias



      :type: [d, m2]

   Methods
   ~~~~~~~

   .. autoapisummary::

      dagma.locally_connected.LocallyConnected.reset_parameters
      dagma.locally_connected.LocallyConnected.forward
      dagma.locally_connected.LocallyConnected.extra_repr

.. toctree::
   :titlesonly:
   :maxdepth: 2
   :hidden:

   reset_parameters<reset_parameters>
   forward<forward>
   extra_repr<extra_repr>

