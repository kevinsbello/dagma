:py:class:`dagma.locally_connected.LocallyConnected <dagma.locally_connected.LocallyConnected>`
===============================================================================================

.. _dagma.locally_connected.LocallyConnected:

.. py:class:: dagma.locally_connected.LocallyConnected(num_linear: int, input_features: int, output_features: int, bias: bool = True)


   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   Local linear layer, i.e. Conv1dLocal() with filter size 1.

   :param num_linear: num of local linear layers, i.e.
   :type num_linear: int
   :param input_features: m1
   :type input_features: int
   :param output_features: m2
   :type output_features: int
   :param bias: Whether to include bias or not. Default: ``True``.
   :type bias: bool, optional
   :param Shape:
   :param -----:
   :param input:
   :type input: [n, d, m1]
   :param output:
   :type output: [n, d, m2]

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

