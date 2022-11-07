# DAGMA

This is an implementation of the following paper:

[1] Bello K., Aragam B., Ravikumar P. (2022). [DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization](https://arxiv.org/abs/2209.08037) ([NeurIPS 2022](https://nips.cc/Conferences/2022/)). 

[notears]: https://arxiv.org/abs/1803.01422

If you find this code useful, please consider citing:
```
@inproceedings{bello2022dagma,
    author = {Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization}},
    year = {2022}
}
```

## TL;DR

We propose a new acyclicity characterization of DAGs via a log-det function for learning DAGs from observational data. Similar to previously proposed acyclicity functions (e.g. [NOTEARS][notears]), our characterization is also exact and differentiable. However, when compared to existing characterizations, our log-det function: (1) Is better at detecting large cycles; (2) Has better behaved gradients; and (3) Its runtime is in practice about an order of magnitude faster. This advantages of our log-det formulation leads to significant improvements in structure accuracy (e.g. SHD).


## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `python-igraph`
- `torch`: Only used for nonlinear models.

## Contents 

- `dagma_linear.py` - implementation of DAGMA for linear models with l1 regularization (supports L2 and Logistic losses).
- `dagma_nonlinear.py` - implementation of DAGMA for nonlinear models using MLP
- `locally_connected.py` - special layer structure used for MLP
- `utils.py` - graph simulation, data simulation, and accuracy evaluation


## Running DAGMA

Use `requirements.txt` to install the dependencies (recommended to use virtualenv or conda).
The simplest way to try out DAGMA is to run a simple example:
```bash
$ git clone https://github.com/kevinsbello/dagma.git
$ pip3 install -r requirements.txt
$ python3 dagma_linear.py
```

The above runs the L1-regularized DAGMA on a randomly generated 20-node Erdos-Renyi graph with 500 samples. 
Within a few seconds, you should see an output like this:
```
{'fdr': 0.0, 'tpr': 1.0, 'fpr': 0.0, 'shd': 0, 'nnz': 20}
```
The data, ground truth graph, and the estimate will be stored in `X.csv`, `W_true.csv`, and `W_est.csv`. 


## Acknowledgments

We thank the authors of the [NOTEARS repo][notears-repo] for making their code available. Part of our code is based on their implementation, specially the `utils.py` file and some code from their implementation of nonlinear models.

[notears-repo]: https://github.com/xunzheng/notears