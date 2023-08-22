# DAGMA: Faster and more accurate structure learning with a log-det constraint

This is an implementation of the following paper:

[1] Bello K., Aragam B., Ravikumar P. (2022). [DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization][dagma]. [NeurIPS'22](https://nips.cc/Conferences/2022/). 

[notears]: https://arxiv.org/abs/1803.01422
[dagma]: https://arxiv.org/abs/2209.08037

If you find this code useful, please consider citing:
```
@inproceedings{bello2022dagma,
    author = {Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization}},
    year = {2022}
}
```

## Installing DAGMA

We recommend using a virtual environment via `virtualenv` or `conda`, and use `pip` to install the `dagma` package.
```bash
$ pip install dagma
```

## Using DAGMA

See an example on how to use dagma in this [iPython notebook][example].

## Table of Content
  * [Summary](#summary)
    + [The log-det acyclicity characterization](#the-log-det-acyclicity-characterization)
    + [A path-following approach](#a-path-following-approach)
  * [Requirements](#requirements)
  * [Contents](#contents)
  * [Acknowledgments](#acknowledgments)

## Summary

We propose a new acyclicity characterization of DAGs via a log-det function for learning DAGs from observational data. Similar to previously proposed acyclicity functions (e.g. [NOTEARS][notears]), our characterization is also exact and differentiable. However, when compared to existing characterizations, our log-det function: (1) Is better at detecting large cycles; (2) Has better-behaved gradients; and (3) Its runtime is in practice about an order of magnitude faster. These advantages of our log-det formulation, together with a path-following scheme, lead to significant improvements in structure accuracy (e.g. SHD).

### The log-det acyclicity characterization

Let $W \in \mathbb{R}^{d\times d}$ be a weighted adjacency matrix  of a graph of $d$ nodes, the log-det function takes the following form:

$$h^{s}(W) = -\log \det (sI-W\circ W) + d \log s,$$

where $I$ is the identity matrix, $s$ is a given scalar (e.g., 1), and $\circ$ denotes the element-wise Hadamard product. Of particular interest, we have that $h(W) = 0$ if and only if $W$ represents a DAG, and when the domain of $h$ is the set of M-matrices then $h$ is well-defined and non-negative. For more properties of $h(W)$ (e.g., being an invex function), $\nabla h(W)$, and $\nabla^2 h(W)$, we invite you to look at [[1][dagma]].

### A path-following approach

Given the exact differentiable characterization of a DAG, we are interested in solving the following optimization problem:
```math
\begin{array}{cl}
\min _{W \in \mathbb{R}^{d \times d}} & Q(W;\mathbf{X}) \\
\text { subject to } & h^{s}(W) = 0,
\end{array}
```
where $Q$ is a given score function (e.g., square loss) that depends on $W$ and the dataset $\mathbf{X}$. To solve the above constrained problem, we propose a path-following approach where we solve a few of the following unconstrained problems:
```math
\hat{W}^{(t+1)} = \arg\min_{W}\; \mu^{(t)} Q(W;\mathbf{X}) + h(W),
```
where $\mu^{(t)} \to 0$ as $t$ increases. Leveraging the properties of $h$, we show that, at the limit, the solution is a DAG. The trick to make this work is to **use the previous solution as a starting point when solving the current unconstrained problem**, as usually done in interior-point algorithms. Finally, we use a simple accelerated gradient descent method to solve each unconstrained problem.

Let us give an illustration of how DAGMA works in a two-node graph (see Figure 1 in [[1][dagma]] for more details). Here $w_1$ (the x-axis) represents the edge weight from node 1 to node 2; while $w_2$ (y-axis) represents the edge weight from node 2 to node 1. Moreover, in this example, the ground-truth DAG corresponds to $w_1 = 1.2$ and $w_2 = 0$. 

Below we have 4 plots, where each illustrates the solution to an unconstrained problem for different values of $\mu$. In the top-left plot, we have $\mu=1$, and we solve the unconstrained problem starting at the empty graph (i.e., $w_1 = w_2 = 0$), which corresponds to the red point, and after running gradient descent, we arrive at the cyan point (i.e., $w_1 = 1.06, w_2 = 0.24$). Then, for the next unconstrained problem in the top-right plot, we have $\mu = 0.1$, and we initialize gradient descent at the previous solution, i.e., $w_1 = 1.06, w_2 = 0.24$, and arrive at the cyan point $w_1 = 1.16, w_2 = 0.04$. Similarly, DAGMA solves for $\mu=0.01$ and $\mu=0.001$, and we can observe how the solution at the final iteration (bottom-right plot) is close to the ground-truth DAG  $w_1 = 1.2, w_2 = 0$.

<img width="1350" alt="dagma_4iters" src="https://user-images.githubusercontent.com/6846921/200969570-8a3434d5-b3b8-4260-966b-6fe1e0303188.png">


## Requirements

- Python 3.7+
- `numpy`
- `scipy`
- `igraph`
- `tqdm`
- `torch`: Only used for nonlinear models.

## Contents 

- `linear.py` - implementation of DAGMA for linear models with l1 regularization (supports L2 and Logistic losses).
- `nonlinear.py` - implementation of DAGMA for nonlinear models using MLP
- `locally_connected.py` - special layer structure used for MLP
- `utils.py` - graph simulation, data simulation, and accuracy evaluation


## Acknowledgments

We thank the authors of the [NOTEARS repo][notears-repo] for making their code available. Part of our code is based on their implementation, specially the `utils.py` file and some code from their implementation of nonlinear models.

[notears-repo]: https://github.com/xunzheng/notears
[example]: https://github.com/kevinsbello/dagma/blob/main/examples/dagma_test.ipynb