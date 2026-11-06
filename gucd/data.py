"""Networks, observed pairs and memberships."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def network_from_od(origin, destination, trips, n):
    """A = OD + OD^T without the diagonal: trips between block groups u and v in either direction. origin and
    destination are node indices 0..n-1."""
    od = sp.csr_matrix((np.asarray(trips, float), (np.asarray(origin), np.asarray(destination))), shape=(n, n))
    A = (od + od.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def load_network(path):
    """Symmetric adjacency matrix saved with scipy.sparse.save_npz, as float64 CSR with zero diagonal."""
    A = sp.load_npz(path).tocsr().astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    if A.shape[0] and abs(A - A.T).max() != 0:
        raise ValueError("the network must be symmetric (A = OD + OD^T)")
    return A


def load_memberships(path):
    """data/memberships/<network>.csv.gz -> (GISJOIN identifiers, F as an N x K array; NaN rows for block groups
    without trips to other block groups of their network)."""
    d = pd.read_csv(path, dtype={"GISJOIN": str})
    cols = [c for c in d.columns if c.startswith("F_")]
    return d["GISJOIN"].to_numpy(), d[cols].to_numpy(float)


def observed_pairs(A, valid=None):
    """Ordered pairs (u, v) with trips, both endpoints valid, sorted by (u, v), and y = log A_uv. These are the
    pairs on which every representation is evaluated (L1). valid defaults to the block groups with trips."""
    A = A.tocsr()
    if valid is None:
        valid = np.diff(A.indptr) > 0
    coo = A.tocoo()
    r, c, w = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data
    keep = valid[r] & valid[c]
    r, c, w = r[keep], c[keep], w[keep]
    o = np.lexsort((c, r))
    r, c, w = r[o], c[o], w[o]
    return dict(rows=r, cols=c, y=np.log(w), valid=np.asarray(valid, bool))
