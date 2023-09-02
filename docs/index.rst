.. image:: ../logo/dagma.png
   :align: center


The :obj:`dagma` library is a Python 3 package for learning DAGs (a.k.a. Bayesian networks) from data.

DAGMA works by optimizing a given ``score/loss function``, where the structure that relates the variables
is constrained to be a ``directed acyclic graph`` (DAG).
Due to the super-exponential number of DAGs w.r.t. the number of variables, 
the vanilla formulation results in a hard combinatorial optimization problem. 
DAGMA reformulates this optimization problem, by replacing the combinatorial constraint
with a non-convex differentiable function that exactly characterizes DAGs, thus, 
making the optimization amenable to continuous optimization methods such as gradient descent. 


.. important::

   If this library was useful in your research, please consider citing us. 

   [1] Bello K., Aragam B., Ravikumar P. (2022). 
   `DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization. <https://arxiv.org/abs/2209.08037>`_ 
   Neural Information Processing Systems (NeurIPS).

   .. md-tab-set::

      .. md-tab-item:: BibTeX

         .. code-block:: latex

            @inproceedings{bello2022dagma,
               author = {Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
               booktitle = {Advances in Neural Information Processing Systems},
               title = {{DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization}},
               year = {2022}
            }

.. note::

   This project is under active development. If you encounter any issues, please raise the issue in the `GitHub page <https://github.com/kevinsbello/dagma>`_.

Features
--------

- Supports continuous data for linear (see :py:mod:`dagma.linear`) and nonlinear models (see :py:mod:`dagma.nonlinear`).
- Supports binary (0/1) data for generalized linear models, via :py:class:`~dagma.linear.DagmaLinear` and using ``logistic`` as score.
- Faster than other continuous optimization methods for structure learning, e.g., NOTEARS, GOLEM.

A Quick Overview of DAGMA
-------------------------

We propose a new acyclicity characterization of DAGs via a log-det function for learning DAGs from observational data. Similar to previously proposed acyclicity functions (e.g. [NOTEARS][notears]), our characterization is also exact and differentiable. However, when compared to existing characterizations, our log-det function: (1) Is better at detecting large cycles; (2) Has better-behaved gradients; and (3) Its runtime is in practice about an order of magnitude faster. These advantages of our log-det formulation, together with a path-following scheme, lead to significant improvements in structure accuracy (e.g. SHD).

The log-det acyclicity characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`W \in \mathbb{R}^{d\times d}` be a weighted adjacency matrix  of a graph of :math:`d` nodes, the log-det function takes the following form:
   .. math::

      h^{s}(W) = -\log \det (sI - W \circ W) + d \log s,
      
where :math:`I`` is the identity matrix, :math:`s` is a given scalar (e.g., 1), and :math:`\circ` denotes the element-wise Hadamard product. 
Of particular interest, we have that :math:`h(W) = 0` *if and only if* :math:`W` represents a DAG, and when the domain of 
:math:`h` is the set of M-matrices then :math:`h` is well-defined and non-negative. 
For more properties of :math:`h(W)` (e.g., being an invex function), :math:`\nabla h(W)`, and :math:`\nabla^2 h(W)`, we invite you to look at 
our `paper <https://arxiv.org/abs/2209.08037>`_.

A path-following approach
~~~~~~~~~~~~~~~~~~~~~~~~~

Given the exact differentiable characterization of a DAG, we are interested in solving the following optimization problem:
   .. math::

      \begin{array}{cl}
      \min _{W \in \mathbb{R}^{d \times d}} & Q(W;\mathbf{X}) \\
      \text { subject to } & h^{s}(W) = 0,
      \end{array}

where :math:`Q` is a given score function (e.g., square loss) that depends on :math:`W` and the dataset :math:`\mathbf{X}`. 
To solve the above constrained problem, we propose a path-following approach where we solve a few of the following unconstrained problems:
   .. math::

      \hat{W}^{(t+1)} = \arg\min_{W}\; \mu^{(t)} Q(W;\mathbf{X}) + h(W),

where :math:`\mu^{(t)} \to 0` as :math:`t` increases. 
Leveraging the properties of :math:`h`, we show that, at the limit, the solution is a DAG. 
The trick to make this work is to **use the previous solution as a starting point when solving the current unconstrained problem**, 
as usually done in interior-point algorithms. Finally, we use a simple accelerated gradient descent method to solve each unconstrained problem.

Let us give an illustration of how DAGMA works in a two-node graph (see Figure 1 in our `paper <https://arxiv.org/abs/2209.08037>`_ for more details). 
Here :math:`w_1` (the x-axis) represents the edge weight from node 1 to node 2; while :math:`w_2` (y-axis) 
represents the edge weight from node 2 to node 1. Moreover, in this example, the ground-truth DAG corresponds to :math:`w_1 = 1.2` and :math:`w_2 = 0`. 

Below we have 4 plots, where each illustrates the solution to an unconstrained problem for different values of :math:`\mu`. 
In the top-left plot, we have :math:`\mu=1`, and we solve the unconstrained problem starting at the empty graph (i.e., :math:`w_1 = w_2 = 0`), 
which corresponds to the red point, and after running gradient descent, we arrive at the cyan point (i.e., :math:`w_1 = 1.06, w_2 = 0.24`). 
Then, for the next unconstrained problem in the top-right plot, we have :math:`\mu = 0.1`, 
and we initialize gradient descent at the previous solution, i.e., :math:`w_1 = 1.06, w_2 = 0.24`, 
and arrive at the cyan point :math:`w_1 = 1.16, w_2 = 0.04`. 
Similarly, DAGMA solves for :math:`\mu=0.01` and :math:`\mu=0.001`, 
and we can observe how the solution at the final iteration (bottom-right plot) is close to the ground-truth DAG  :math:`w_1 = 1.2, w_2 = 0`.

.. image:: https://user-images.githubusercontent.com/6846921/200969570-8a3434d5-b3b8-4260-966b-6fe1e0303188.png
   :width: 1350
   :alt: dagma_4iters
   :align: center

.. toctree::
   :caption: Home
   :hidden:

   Home<self>

.. toctree::
   :caption: Getting Started
   :hidden:

   usage

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2
   :titlesonly:

   api/dagma/index

