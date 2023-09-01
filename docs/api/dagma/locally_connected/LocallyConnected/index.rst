:py:class:`dagma.locally_connected.LocallyConnected`
==================================================

.. _dagma.locally_connected.LocallyConnected:

.. py:class:: LocallyConnected(num_linear, input_features, output_features, bias=True)


   Bases: :py:obj:`torch.nn.Module`

   Local linear layer, i.e. Conv1dLocal() with filter size 1.

   :param num_linear: num of local linear layers, i.e.
   :param input_features: m1
   :param output_features: m2
   :param bias: whether to include bias or not

   Shape:
       - Input: [n, d, m1]
       - Output: [n, d, m2]

   .. attribute:: weight

      [d, m1, m2]

   .. attribute:: bias

      [d, m2]

   Initializes internal Module state, shared by both nn.Module and ScriptModule.

   Methods
   ~~~~~~~

   .. autoapisummary::

      dagma.locally_connected.LocallyConnected.reset_parameters
      dagma.locally_connected.LocallyConnected.forward
      dagma.locally_connected.LocallyConnected.extra_repr

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :hidden:

   reset_parameters<reset_parameters>
   forward<forward>
   extra_repr<extra_repr>
