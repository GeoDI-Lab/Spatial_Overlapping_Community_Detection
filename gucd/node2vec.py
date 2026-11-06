"""node2vec with p = q = 1 (weighted DeepWalk): random walks in numpy, skip-gram with negative sampling in PyTorch.

  walks      first-order weighted walks, P(u -> v) = A_uv / sum_w A_uw; 10 walks per node of length 80
  skip-gram  separate input and output embeddings; input ~ U(-0.5/d, 0.5/d), output 0 (as in word2vec)
             dynamic window: for each centre position b ~ U{1..10}, contexts at offsets 1..b on both sides
             5 negatives per positive pair from the unigram distribution^0.75 of the walks; no subsampling
             loss -log s(w_u.c_v) - sum_k log s(-w_u.c_nk), mean over mini-batches of 4,096 positive pairs
             SparseAdam, learning rate 0.01 decaying linearly over 5 epochs (walk order and windows redrawn per epoch)
  output     the input vectors W
"""
import time
import numpy as np

DEFAULTS = dict(walks_per_node=10, walk_length=80, window=10, negative=5, epochs=5, lr=0.01, batch=4096,
                chunk_walks=2048, ns_exponent=0.75)


def weighted_walks(A, starts, walk_length, rng):
    """First-order weighted random walks on a csr matrix A (sorted indices, no zero rows among visited nodes).
    Returns int64 array (len(starts), walk_length)."""
    indptr, indices, data = A.indptr, A.indices, A.data.astype(np.float64)
    cum = np.cumsum(data)
    before = np.concatenate([[0.0], cum])[indptr[:-1]]          # cumulative weight before each row
    s = np.asarray(A.sum(1)).ravel()
    cur = np.asarray(starts, np.int64)
    if (s[cur] <= 0).any():
        raise ValueError("walk starts at an isolated node")
    W = np.empty((len(cur), walk_length), np.int64)
    W[:, 0] = cur
    for t in range(1, walk_length):
        target = before[cur] + rng.random(len(cur)) * s[cur]
        pos = np.searchsorted(cum, target, side="right")
        pos = np.clip(pos, indptr[cur], indptr[cur + 1] - 1)
        cur = indices[pos].astype(np.int64)
        W[:, t] = cur
    return W


def window_pairs(walks, window, rng):
    """(centre, context) pairs with word2vec's dynamic window: reduced window b ~ U{1..window} per centre position,
    contexts at offsets 1..b on both sides."""
    n, L = walks.shape
    b = rng.integers(1, window + 1, size=(n, L))
    cs, xs = [], []
    for k in range(1, min(window, L - 1) + 1):
        m = b[:, :L - k] >= k                  # centre at i, context at i + k
        cs.append(walks[:, :L - k][m])
        xs.append(walks[:, k:][m])
        m = b[:, k:] >= k                      # centre at i (>= k), context at i - k
        cs.append(walks[:, k:][m])
        xs.append(walks[:, :L - k][m])
    return np.concatenate(cs), np.concatenate(xs)


def train_node2vec(A, d, seed=0, log=None, threads=2, **kw):
    """A: csr (n x n) weighted undirected, every node with degree > 0 (caller removes isolated nodes).
    Returns (W: n x d float64, info dict)."""
    import torch
    import torch.nn.functional as TF
    torch.set_num_threads(threads)
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    starts = np.tile(np.arange(n), cfg["walks_per_node"])
    walks = weighted_walks(A, starts, cfg["walk_length"], rng)
    t_walk = time.time() - t0
    counts = np.bincount(walks.ravel(), minlength=n).astype(np.float64)
    noise = torch.tensor(counts ** cfg["ns_exponent"], dtype=torch.float64)
    noise = (noise / noise.sum()).float()

    Win = torch.nn.Embedding(n, d, sparse=True)
    Cout = torch.nn.Embedding(n, d, sparse=True)
    with torch.no_grad():
        Win.weight.uniform_(-0.5 / d, 0.5 / d)
        Cout.weight.zero_()
    opt = torch.optim.SparseAdam(list(Win.parameters()) + list(Cout.parameters()), lr=cfg["lr"])
    E, B, neg, chunk = cfg["epochs"], cfg["batch"], cfg["negative"], cfg["chunk_walks"]
    nw = len(walks)
    loss_epoch, pairs_epoch, eval_epoch = [], [], []
    # convergence diagnostic that does not affect training: the loss on a fixed sample of pairs and negatives,
    # before training and after every epoch (own random generator, so the training stream is unchanged)
    erng = np.random.default_rng(10_000 + seed)
    ec, ex = window_pairs(walks[erng.choice(nw, min(nw, 2000), replace=False)], cfg["window"], erng)
    esel = erng.choice(len(ec), min(len(ec), 200_000), replace=False)
    ec_t, ex_t = torch.from_numpy(ec[esel]), torch.from_numpy(ex[esel])
    en_t = torch.from_numpy(erng.choice(n, size=(len(esel), neg), p=(counts ** cfg["ns_exponent"]) /
                                        (counts ** cfg["ns_exponent"]).sum()))

    def eval_loss():
        with torch.no_grad():
            wu = Win(ec_t)
            pos = TF.logsigmoid((wu * Cout(ex_t)).sum(-1))
            ng = TF.logsigmoid(-torch.bmm(Cout(en_t), wu.unsqueeze(-1)).squeeze(-1)).sum(-1)
            return float(-(pos + ng).mean())
    eval_epoch.append(eval_loss())
    n_steps = 0
    t1 = time.time()
    for ep in range(E):
        order = rng.permutation(nw)
        tot, npair = 0.0, 0
        for c0 in range(0, nw, chunk):
            idx = order[c0:c0 + chunk]
            cen, ctx = window_pairs(walks[idx], cfg["window"], rng)
            perm = rng.permutation(len(cen))
            cen_t = torch.from_numpy(cen[perm])
            ctx_t = torch.from_numpy(ctx[perm])
            lr = cfg["lr"] * max(1.0 - (ep + (c0 + len(idx)) / nw) / E, 1e-4)
            for g in opt.param_groups:
                g["lr"] = lr
            for s0 in range(0, len(cen_t), B):
                u = cen_t[s0:s0 + B]
                v = ctx_t[s0:s0 + B]
                negs = torch.multinomial(noise, len(u) * neg, replacement=True).view(len(u), neg)
                wu = Win(u)
                pos = TF.logsigmoid((wu * Cout(v)).sum(-1))
                ng = TF.logsigmoid(-torch.bmm(Cout(negs), wu.unsqueeze(-1)).squeeze(-1)).sum(-1)
                loss = -(pos + ng).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(u)
                npair += len(u)
                n_steps += 1
        loss_epoch.append(tot / max(npair, 1))
        pairs_epoch.append(npair)
        eval_epoch.append(eval_loss())
        if log:
            log(f"    epoch {ep + 1}/{E}: mean SGNS loss {loss_epoch[-1]:.5f}, fixed-sample loss {eval_epoch[-1]:.5f} "
                f"({npair} pairs, lr end {lr:.2e}, {time.time() - t1:.0f}s)")
    W = Win.weight.detach().double().numpy().copy()
    info = dict(n2v_d=d, n2v_seed=seed, n2v_n_nodes=n, n2v_n_walks=int(nw), n2v_tokens=int(walks.size),
                n2v_pairs_per_epoch_mean=float(np.mean(pairs_epoch)), n2v_steps=int(n_steps),
                n2v_loss_epochs=json_list(loss_epoch), n2v_loss_last=float(loss_epoch[-1]),
                n2v_loss_rel_drop_last_epoch=float((loss_epoch[-2] - loss_epoch[-1]) / loss_epoch[-2]) if E > 1 else
                float("nan"),
                n2v_evalloss_epochs=json_list(eval_epoch), n2v_evalloss_last=float(eval_epoch[-1]),
                n2v_evalloss_rel_drop_last_epoch=float((eval_epoch[-2] - eval_epoch[-1]) / eval_epoch[-2]),
                n2v_t_walks=t_walk, n2v_t_train=time.time() - t1,
                n2v_cfg=json_list([f"{k}={v}" for k, v in cfg.items()]))
    return W, info


def json_list(x):
    import json
    return json.dumps([float(v) if isinstance(v, (float, np.floating)) else v for v in x])


def embed_network(A, valid=None, d=128, seed=0, threads=2, log=None):
    """node2vec embedding of a network as used for the input of the GCN: trained on A restricted to the valid block
    groups with trips; returns W (N x d) with NaN rows for the other block groups, and the training information."""
    A = A.tocsr()
    N = A.shape[0]
    valid = np.ones(N, bool) if valid is None else np.asarray(valid, bool)
    import scipy.sparse as sp
    v = valid.astype(float)
    Av = sp.csr_matrix(sp.diags(v) @ A @ sp.diags(v))
    Av.eliminate_zeros()
    Av.sort_indices()
    emb = valid & (np.diff(Av.indptr) > 0)
    idx = np.where(emb)[0]
    sub = Av[idx][:, idx].tocsr()
    sub.sort_indices()
    W, info = train_node2vec(sub, d, seed=seed, log=log, threads=threads)
    Wfull = np.full((N, d), np.nan)
    Wfull[idx] = W
    return Wfull, info
