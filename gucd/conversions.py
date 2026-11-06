"""Flow models of a representation, fitted under the same error as F (Fig. 2a).

Every representation is converted into a flow model whose weights minimize the L1 on the observed pairs (least
absolute deviations, solved exactly as a linear programme): F itself enters as F_u . F_v; its binary version
(B_uk = 1 where F_uk > 0.5) as a + sum_k w_k B_uk B_vk; its hard partition (c_u = argmax_k F_uk) as a free
symmetric block matrix, Omega[c_u, c_v], whose least-absolute-deviation fit is the median log flow of each block.
Because A and every model are symmetric, the L1 over ordered pairs equals the L1 over unordered pairs.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog


def lad_lp(X, y, wts=None):
    """Weighted least absolute deviations, min sum_i wts_i |y_i - X_i theta|, as an exact HiGHS linear programme."""
    n, p = X.shape
    if wts is None:
        wts = np.ones(n)
    cost = np.concatenate([np.zeros(p), wts, wts])
    Aeq = sp.hstack([sp.csr_matrix(X), sp.identity(n, format="csr"), -sp.identity(n, format="csr")]).tocsc()
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(cost, A_eq=Aeq, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:p], float(res.fun / wts.sum())


def lad_binary_design(Xb, y):
    """Least absolute deviations with an intercept on a 0/1 design (rows with the same pattern and y are merged into
    weights first). Returns (intercept, weights, L1)."""
    n, k = Xb.shape
    code = (Xb.astype(np.int64) * (1 << np.arange(k, dtype=np.int64))).sum(1)
    g = pd.DataFrame(dict(code=code, y=np.round(y, 10))).groupby(["code", "y"], sort=False).size().reset_index(name="cnt")
    codes = g["code"].to_numpy()
    Xc = np.hstack([np.ones((len(g), 1)), ((codes[:, None] >> np.arange(k)) & 1).astype(float)])
    used = np.r_[True, Xc[:, 1:].any(0)]              # a community absent from all pairs is not identifiable
    theta = np.zeros(k + 1)
    th, L1 = lad_lp(Xc[:, used], g["y"].to_numpy(), g["cnt"].to_numpy().astype(float))
    theta[used] = th
    return theta[0], theta[1:], L1


def block_medians(za, zb, y, Kb):
    """Omega[a, b] = median log flow over the pairs with {z_u, z_v} = {a, b} (NaN for empty blocks)."""
    a, b = np.minimum(za, zb), np.maximum(za, zb)
    s = pd.Series(y).groupby(a.astype(np.int64) * Kb + b).median()
    Om = np.full((Kb, Kb), np.nan)
    Om[s.index.values // Kb, s.index.values % Kb] = s.values
    Om[s.index.values % Kb, s.index.values // Kb] = s.values
    return Om


def _unordered(pairs):
    up = pairs["rows"] < pairs["cols"]
    return pairs["rows"][up], pairs["cols"][up], pairs["y"][up]


def l1_weighted(F, pairs):
    """L1 of F itself: mean |F_u . F_v - log A_uv| over the observed pairs."""
    F = np.nan_to_num(np.asarray(F, float))
    pred = np.einsum("ij,ij->i", F[pairs["rows"]], F[pairs["cols"]])
    return float(np.mean(np.abs(pred - pairs["y"])))


def l1_binary(F, pairs, threshold=0.5):
    """L1 of the binary version of F (memberships F > threshold, an intercept and one fitted weight per community)."""
    B = np.nan_to_num(np.asarray(F, float)) > threshold
    r, c, y = _unordered(pairs)
    a, w, L1 = lad_binary_design(B[r] & B[c], y)
    return L1, dict(intercept=a, weights=w)


def l1_partition(F, pairs):
    """L1 of the hard partition of F (each block group in its community of largest membership; all-zero rows in
    community 1) with a fitted symmetric block matrix."""
    F = np.nan_to_num(np.asarray(F, float))
    z = F.argmax(1)
    r, c, y = _unordered(pairs)
    Om = block_medians(z[r], z[c], y, F.shape[1])
    pred = Om[z[r], z[c]]
    return float(np.mean(np.abs(y - pred))), dict(block_matrix=Om)
