DAGMA
===================================

**DAGMA** is a Python library for learning DAGs (a.k.a. Bayesian networks)
by optimizing a given score/loss function contrained by the DAG structure.
Due to the super-exponential number of DAGs w.r.t. the number of variables, 
the vanilla formulation is a hard combinatorial optimization problem. 
DAGMA tackles this optimization problem, by replacing the combinatorial constraint
by a non-convex differentiable function that characterizes DAGs, thus, 
making the optimization more tractable. 

.. attention::
   This is an implementation of the following paper:

   [1] Bello K., Aragam B., Ravikumar P. (2022). `DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization. <https://arxiv.org/abs/2209.08037>`_ NeurIPS'22.

   If you find this code useful, please consider citing:

   .. code-block:: tex
      
      @inproceedings{bello2022dagma,
      author = {Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
      booktitle = {Advances in Neural Information Processing Systems},
      title = {{DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization}},
      year = {2022}
      }

Check out the :doc:`usage` section for further information, including
how to do :ref:`installation` of the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   Home <self>
   usage
   api
