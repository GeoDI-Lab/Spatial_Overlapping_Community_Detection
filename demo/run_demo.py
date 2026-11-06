#!/usr/bin/env python
"""Demo on New York (6,593 census block groups; the network is in demo/data).

1. Loads the network and the published memberships F (data/memberships/New York.csv.gz).
2. Reproduces New York's reconstruction errors of Fig. 2a: L1 of F, of its binary version and of its hard
   partition, and compares them with the paper's values (data/results/figure2/source_data_Figure2_b.csv).
3. Summarizes the overlap count and the overlap zone.
4. With --refit: infers F anew with one training seed (node2vec input + GCN), as scripts/fit_memberships.py does for
   seeds 0-4. This takes about 15-40 minutes on a laptop CPU for New York, the largest of the 50 networks.

  python demo/run_demo.py [--refit] [--seed 0]
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gucd import conversions, data, model, node2vec, overlap  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A = data.load_network(os.path.join(ROOT, "demo", "data", "A_New York.npz"))
    gis, F = data.load_memberships(os.path.join(ROOT, "data", "memberships", "New York.csv.gz"))
    valid = ~np.isnan(F).any(1)
    pairs = data.observed_pairs(A, valid)
    print(f"New York: {A.shape[0]} block groups, {valid.sum()} with trips to others, "
          f"{len(pairs['y']) // 2} pairs of block groups with trips, {A.sum() / 2:,.0f} trips between block groups")

    ref = pd.read_csv(os.path.join(ROOT, "data", "results", "figure2", "source_data_Figure2_b.csv")).set_index("city")
    rows = [("Weighted overlap F", conversions.l1_weighted(F, pairs), "Weighted overlap F (this study)"),
            ("Binary version (F > 0.5)", conversions.l1_binary(F, pairs)[0], "Binary overlap"),
            ("Hard partition", conversions.l1_partition(F, pairs)[0], "Hard partition")]
    print("\nReconstruction error L1 (mean |predicted - observed log flow| over the observed pairs)")
    for name, got, col in rows:
        print(f"  {name:26s} {got:.6f}   paper {ref.loc['New York', 'L1 | ' + col]:.6f}")

    cnt = overlap.overlap_count(F)
    zone, rest = overlap.overlap_zone(F)
    print(f"\nOverlap count (communities with F > 0.5): {np.bincount(cnt[cnt >= 0]).tolist()} block groups with 0, 1, 2, ...")
    print(f"Overlap zone: {zone.sum()} of {zone.sum() + rest.sum()} block groups with membership")

    if a.refit:
        t = time.time()
        print(f"\nRefitting F with seed {a.seed} (node2vec, then the GCN)...")
        W, _ = node2vec.embed_network(A, valid, d=128, seed=0)
        r = model.fit_memberships(A, model.standardize_embedding(W), seed=a.seed,
                                  log=lambda s: print("  " + s, flush=True))
        L1 = conversions.l1_weighted(r["F"], pairs)
        print(f"  L1 of the refitted F: {L1:.6f} ({r['steps']} steps, {time.time() - t:.0f} s). The paper's F is the "
              f"best of seeds 0-4 (data/memberships/manifest.csv).")


if __name__ == "__main__":
    main()
