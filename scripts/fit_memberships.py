#!/usr/bin/env python
"""Infer the memberships F of one network as in the paper: node2vec input (d = 128, seed 0), the GCN fitted with
seeds 0-4, and F = the seed with the lowest L1 on the observed pairs (ties: the smaller seed).

  python scripts/fit_memberships.py --network A_City.npz --out F_City.csv.gz [--seeds 0 1 2 3 4] [--K 10]
         [--valid memberships.csv.gz]

--network  symmetric adjacency matrix saved with scipy.sparse.save_npz (trips between block groups, A = OD + OD^T)
--valid    optional: a memberships file whose empty rows mark block groups to leave out (as in the paper); by
           default every block group with trips is used
Output: CSV with node index and F_1 ... F_K (empty rows for block groups left out), plus <out>.seeds.csv with the L1
of every seed. Two threads and deterministic algorithms make the fits reproducible on the same platform.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"                                  # before numpy and torch are imported

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gucd import conversions, data, model, node2vec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=model.K_DEFAULT)
    ap.add_argument("--valid")
    a = ap.parse_args()

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)

    A = data.load_network(a.network)
    N = A.shape[0]
    if a.valid:
        _, Fv = data.load_memberships(a.valid)
        valid = ~np.isnan(Fv).any(1)
    else:
        valid = np.diff(A.indptr) > 0
    pairs = data.observed_pairs(A, valid)
    log(f"N = {N} block groups, {valid.sum()} used, {len(pairs['y'])} ordered pairs with trips")
    W, _ = node2vec.embed_network(A, valid, d=128, seed=0, log=log)
    X = model.standardize_embedding(W)
    fits = []
    for s in a.seeds:
        r = model.fit_memberships(A, X, seed=s, K=a.K)
        L1 = conversions.l1_weighted(r["F"], pairs)
        log(f"seed {s}: L1 {L1:.6f} ({r['steps']} steps; {r['stop_reason']})")
        fits.append((L1, s, r["F"]))
    L1, s, F = min(fits, key=lambda x: (x[0], x[1]))
    F = np.where(valid[:, None], F, np.nan)
    out = pd.DataFrame(F, columns=[f"F_{k + 1}" for k in range(a.K)])
    out.insert(0, "node", np.arange(N))
    out.to_csv(a.out, index=False, float_format="%.10g")
    pd.DataFrame([(x[1], x[0]) for x in fits], columns=["seed", "L1"]).to_csv(a.out + ".seeds.csv", index=False)
    log(f"selected seed {s} (L1 {L1:.6f}) -> {a.out}")


if __name__ == "__main__":
    main()
