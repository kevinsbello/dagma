from locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np
from  torch import optim


class DagmaNN(nn.Module):
    
    def __init__(self, dims, bias=True, verbose=False):
        super(DagmaNN, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.d = dims[0]
        self.I = torch.eye(self.d)
        self.vprint = print if verbose else lambda *a, **k: None
        self.dims = dims
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        
    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s=1.0):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        reg += torch.sum(self.fc1.weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n, d = target.shape
    # loss = 0.5 / n * torch.sum((output - target) ** 2)
    loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
    return loss


def minimize(model, X, max_iter, lr, lambda1, lambda2, mu, s, checkpoint=1000, tol=1e-6, verbose=False):
    vprint = print if verbose else lambda *a, **k: None
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.99,.999))#, weight_decay=rho*lambda2)
    obj_prev = 1e16
    for i in range(int(max_iter)):
        optimizer.zero_grad()
        X_hat = model(X)
        score = squared_loss(X_hat, X)
        h_val = model.h_func(s)
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        obj = mu * (score + l2_reg + l1_reg) + 0.5 * h_val * h_val
        # obj = rho * (loss + l1_reg) + h_val
        obj.backward()
        optimizer.step()
        if i % checkpoint == 0 or i == max_iter-1:
            obj_new = obj.item()
            vprint(f"\nInner iteration {i}")
            vprint(f'\th(W(model)): {h_val.item()}')
            vprint(f'\tscore(model): {obj_new}')
            if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                break
            obj_prev = obj_new
    with torch.no_grad():
        h_new = model.h_func(s).item()
    return h_new


def dagma_nonlinear(
        model: nn.Module, X: torch.tensor,
        lambda1=.02, lambda2=.005,
        T=5, mu_init=1., mu_factor=.1,
        warm_iter=int(5e4), max_iter=int(8e4),
        s=1., h_tol=1e-8, w_threshold=0.15,
        lr=0.0002, checkpoint=1000, verbose=False
    ):
    vprint = print if verbose else lambda *a, **k: None
    mu = mu_init
    if type(s) == list:
        if len(s) < T: 
            vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
            s = s + (T - len(s)) * [s[-1]]
    elif type(s) in [int, float]:
        s = T * [s]
    else:
        ValueError("s should be a list, int, or float.") 
    for i in range(int(T)):
        vprint(f'\nDagma iter t={i+1} -- mu: {mu}')
        inner_iter = max_iter if i == T - 1 else warm_iter
        h = minimize(model, X, inner_iter, lr, lambda1, lambda2, mu, s[i], 
                     checkpoint=checkpoint, verbose=verbose)
        mu *= mu_factor
        vprint(f'h: {h}')
        if h <= h_tol:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
    from timeit import default_timer as timer
    import utils
    
    torch.set_default_dtype(torch.double)
    utils.set_random_seed(1)
    torch.manual_seed(1)
    
    n, d, s0, graph_type, sem_type = 2000, 20, 20, 'ER', 'mlp'
    B_true = utils.simulate_dag(d, s0, graph_type)
    X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

    model = DagmaNN(dims=[d, 10, 1], bias=True)
    X_torch = torch.from_numpy(X)
    tstart = timer()
    W_est = dagma_nonlinear(model, X_torch, lambda1=0.02, lambda2=0.005)
    tend = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(f'runtime: {tend-tstart}')
    print(acc)
