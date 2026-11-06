"""Inference of the weighted overlapping memberships F.

Each block group u has memberships F_u >= 0 in K communities, and the log flow between two block groups is
approximated by the inner product of their memberships, log A_uv ~ F_u . F_v. F is the output of a two-layer graph
convolutional network (GCN),

    F = ReLU(P H W2 + b2),   H = BN(ReLU(P X W1 + b1)),   P = D^-1/2 (log(1 + A) + I) D^-1/2,

with 128 hidden units, BN a standardization of each hidden unit over the block groups without learnable parameters,
and X the node2vec embedding of the network (columns standardized). The weights minimize the mean absolute error of
the log flows on the observed pairs plus 1e-6 (||W1||^2 + ||W2||^2), with Adam. The evaluation L1, computed every 10
epochs, sets the learning rate (halved after 300 epochs without a relative improvement of 1e-4, floor 1e-5), the
checkpoint (lowest value) and the stop (relative improvement below 1e-4 over 1,000 epochs; 2,000 to 20,000 epochs).

With two threads and deterministic algorithms, a fit is reproducible bit for bit on the same platform.
"""
import copy
import math

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as Fn

K_DEFAULT = 10
HIDDEN = [128]
CONFIG = dict(lr=1e-2, sched="plateau", dropout=0.0, max_epochs=20000, eval_every=10, stop_window=1000,
              stop_rel_tol=1e-4, min_epochs=2000, plateau_factor=0.5, plateau_patience=300, plateau_min_frac=1e-3,
              weight_decay=1e-6, adj_weighting="log1p")


# ------------------------------------------------------------------------------------------------ inputs
def training_pairs(A):
    """All unordered pairs with trips (u < v) and their log flows, as torch tensors."""
    coo = sp.triu(A, k=1).tocoo()
    ur = torch.from_numpy(coo.row.astype(np.int64))
    uc = torch.from_numpy(coo.col.astype(np.int64))
    uy = torch.from_numpy(np.log(coo.data)).float()
    return ur, uc, uy


def propagation_matrix(A):
    """P = D^-1/2 (log(1 + A) + I) D^-1/2 (scipy CSR)."""
    M = A.tocsr().copy()
    M.eliminate_zeros()
    M.data = np.log1p(M.data)
    adj = (M + sp.diags(np.ones(M.shape[0]))).tocsr()
    deg = np.ravel(adj.sum(1))
    dinv = 1 / np.sqrt(deg)
    return adj.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()


class _SymSpMM(torch.autograd.Function):
    """y = S @ X for a constant symmetric sparse S; the backward pass is S @ g."""

    @staticmethod
    def forward(ctx, X, S):
        ctx.S = S
        return S @ X

    @staticmethod
    def backward(ctx, g):
        return ctx.S @ g, None


def make_prop(adj_sp, dense_threshold=0.08):
    """Propagation as a function of X: a dense product for small or dense networks, sparse CSR otherwise."""
    N = adj_sp.shape[0]
    density = adj_sp.nnz / float(N * N)
    if density >= dense_threshold or N <= 1200:
        D = torch.from_numpy(adj_sp.toarray()).float()
        return (lambda X: D @ X), "dense"
    coo = adj_sp.tocoo()
    S = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data, size=adj_sp.shape,
                                dtype=torch.float32).coalesce().to_sparse_csr()
    return (lambda X: _SymSpMM.apply(X, S)), "sparse_csr"


def standardize_embedding(W):
    """Input features from a node2vec embedding W (N x d, NaN rows for block groups without an embedding): each
    column standardized over the embedded rows (ddof 0; a zero-variance column is only centred); other rows 0."""
    W = np.asarray(W, dtype=np.float64)
    emb = ~np.isnan(W).any(1)
    mu = W[emb].mean(0)
    sd = W[emb].std(0)
    X = np.zeros_like(W)
    X[emb] = (W[emb] - mu) / np.where(sd > 0, sd, 1.0)
    return torch.from_numpy(X).float()


def spectral_features(A, d=64):
    """Alternative input (Supplementary Note 3): the d leading eigenvectors of D0^-1/2 log(1 + A) D0^-1/2
    (D0, the row sums of log(1 + A)), largest-magnitude entry positive, scaled to unit standard deviation."""
    N = A.shape[0]
    d = min(d, N - 2)
    M = A.copy()
    M.data = np.log1p(M.data)
    deg = np.ravel(M.sum(1))
    dinv = np.zeros_like(deg)
    dinv[deg > 0] = 1 / np.sqrt(deg[deg > 0])
    S = M.multiply(dinv[:, None]).multiply(dinv[None, :]).tocsr()
    if N <= 2500:
        w, V = sla.eigh(S.toarray(), subset_by_index=[N - d, N - 1])
    else:
        v0 = np.random.default_rng(0).random(N)
        w, V = spla.eigsh(S, k=d, which="LA", v0=v0, tol=1e-10)
    o = np.argsort(-w)
    V = V[:, o]
    s = np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
    V = V * s[None, :]
    return torch.from_numpy(V / V.std(0, keepdims=True)).float()


# ------------------------------------------------------------------------------------------------ model
class GraphConvolution(nn.Module):
    """P @ (X @ W) + b, with Xavier-uniform W and zero b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)


class GCN(nn.Module):
    """Two graph convolutions (hidden [128] -> K); ReLU and batch standardization without learnable parameters after
    the hidden layer."""

    def __init__(self, in_dim, hidden, K, dropout=0.0, batch_norm=True):
        super().__init__()
        dims = list(hidden) + [K]
        self.layers = nn.ModuleList([GraphConvolution(in_dim, dims[0])])
        for i in range(len(dims) - 1):
            self.layers.append(GraphConvolution(dims[i], dims[i + 1]))
        self.dropout = dropout
        self.batch_norm = batch_norm

    def forward(self, x, prop):
        L = len(self.layers)
        for idx, layer in enumerate(self.layers):
            if self.dropout != 0:
                x = Fn.dropout(x, p=self.dropout, training=self.training)
            x = prop(x @ layer.weight) + layer.bias
            if idx != L - 1:
                x = Fn.relu(x)
                if self.batch_norm:
                    x = Fn.batch_norm(x, None, None, training=True, momentum=0.1, eps=1e-5)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if "bias" not in n]


def l2_reg_loss(model, scale):
    loss = 0.0
    for w in model.get_weights():
        loss = loss + w.pow(2.0).sum()
    return loss * scale


def l1_obj(Z, ur, uc, uy):
    return torch.mean(torch.abs(torch.sum(Z[ur] * Z[uc], dim=1) - uy))


def train(model, x, prop, ur, uc, uy, cfg=CONFIG, log=None):
    """Adam with a plateau schedule on the evaluation L1; returns F at the best checkpoint (evaluation mode) and the
    history (epoch, train L1, evaluation L1, learning rate), the number of steps, the stop reason, the best L1 and
    its epoch."""
    lr0 = cfg["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    pe = max(1, cfg["plateau_patience"] // cfg["eval_every"])
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["plateau_factor"], patience=pe, threshold=1e-4, threshold_mode="rel",
        min_lr=lr0 * cfg["plateau_min_frac"])
    T = cfg["max_epochs"]
    best, best_state, best_epoch = math.inf, None, -1
    bests, hist = [], []
    steps, stop_reason = 0, "max_epochs"
    win = cfg["stop_window"] // cfg["eval_every"]
    tl = float("nan")
    for epoch in range(T + 1):
        if epoch % cfg["eval_every"] == 0 or epoch == T:
            with torch.no_grad():
                model.eval()
                ev = float(l1_obj(Fn.relu(model(x, prop)), ur, uc, uy))
            if ev < best:
                best, best_epoch = ev, epoch
                best_state = copy.deepcopy(model.state_dict())
            bests.append(best)
            cur_lr = opt.param_groups[0]["lr"]
            hist.append((epoch, tl, ev, cur_lr))
            plateau.step(ev)
            if log and epoch % 1000 == 0:
                log(f"epoch {epoch:5d}  train L1 {tl:.5f}  eval L1 {ev:.5f}  best {best:.5f}  lr {cur_lr:.2e}")
            if epoch >= max(cfg["min_epochs"], cfg["stop_window"]) and len(bests) > win:
                prev = bests[-1 - win]
                if prev - best < cfg["stop_rel_tol"] * prev:
                    stop_reason = f"converged (relative improvement < {cfg['stop_rel_tol']} over {cfg['stop_window']} epochs)"
                    break
        if epoch == T:
            break
        model.train()
        opt.zero_grad()
        Z = Fn.relu(model(x, prop))
        loss = l1_obj(Z, ur, uc, uy)
        tl = float(loss.detach())
        (loss + l2_reg_loss(model, cfg["weight_decay"])).backward()
        opt.step()
        steps += 1
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        F_eval = Fn.relu(model(x, prop))
    return F_eval, np.array(hist, float), steps, stop_reason, best, best_epoch


def fit_memberships(A, X, seed=0, K=K_DEFAULT, threads=2, log=None):
    """Fit F for one network.

    A: symmetric CSR matrix of trips between block groups (zero diagonal); X: input features (torch tensor, N x d),
    e.g. standardize_embedding(W) of the node2vec embedding. Returns a dict with F (N x K, float64), the history and
    the configuration. The paper fits seeds 0-4 and keeps the one with the lowest L1 on the observed pairs."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ur, uc, uy = training_pairs(A)
    prop, prop_kind = make_prop(propagation_matrix(A))
    torch.manual_seed(seed)                              # model initialization independent of the feature step
    model = GCN(X.shape[1], HIDDEN, K, dropout=CONFIG["dropout"], batch_norm=True)
    F, hist, steps, stop_reason, best, best_epoch = train(model, X, prop, ur, uc, uy, CONFIG, log)
    cfg = dict(CONFIG, K=K, hidden=HIDDEN, seed=seed, threads=threads, prop=prop_kind, torch=torch.__version__)
    return dict(F=F.double().numpy(), hist=hist, steps=steps, stop_reason=stop_reason, best_eval_L1=best,
                best_epoch=best_epoch, config=cfg)
