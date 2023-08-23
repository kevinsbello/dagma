.. image:: ../logo/dagma.jpeg
   :align: center


The :obj:`dagma` library is a Python 3 package for learning DAGs (a.k.a. Bayesian networks) from data.

DAGMA works by optimizing a given ``score/loss function``, where the structure that relates the variables
is constrained to be a ``directed acyclic graph`` (DAG).
Due to the super-exponential number of DAGs w.r.t. the number of variables, 
the vanilla formulation results in a hard combinatorial optimization problem. 
DAGMA reformulates this optimization problem, by replacing the combinatorial constraint
by a non-convex differentiable function that exactly characterizes DAGs, thus, 
making the optimization ammenable to continuous optimization methods such as gradient descent. 


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


.. toctree::
   :caption: Home
   :hidden:

   <self>

.. toctree::
   :caption: Getting Started
   :hidden:

   usage.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api.rst

