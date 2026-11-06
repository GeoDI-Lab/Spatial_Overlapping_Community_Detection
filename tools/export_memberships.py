#!/usr/bin/env python
"""Export the memberships F used in the paper to data/memberships/.

Input: the folder of the analysis that selected F (one CSV per network with GISJOIN, lat, lon and the ten memberships
0-9 of the selected training seed, and seeds/<network>__s<seed>.csv for seeds 0-4); set GUCD_F_SETS to it.
Output: data/memberships/<network>.csv.gz and seeds/<network>__s<seed>.csv.gz with columns GISJOIN, lat, lon,
F_1 ... F_10 (community k of the paper is column F_k), and manifest.csv. Rows of block groups without trips to other
block groups of their network are empty (NaN).

  GUCD_F_SETS=/path/to/F_sets/gcnn2v python tools/export_memberships.py
"""
import glob
import hashlib
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "memberships")
NAMES = {"Louisville+Jefferson County metro government": "Louisville",
         "Nashville-Davidson metropolitan government": "Nashville",
         "TCMA_work": "Twin Cities weekday", "TCMA_weekend": "Twin Cities weekend"}


def convert(src, dst):
    d = pd.read_csv(src, dtype={"GISJOIN": str})
    d = d.rename(columns={str(k): f"F_{k + 1}" for k in range(10)})
    d.to_csv(dst, index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.10g")
    return d


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    src = os.environ.get("GUCD_F_SETS")
    if not src:
        raise SystemExit("set GUCD_F_SETS to the folder with the selected memberships (<network>.csv and seeds/)")
    man = pd.read_csv(os.path.join(src, "manifest.csv"))
    man = man[man.file_kind == "primary"].set_index("net")
    os.makedirs(os.path.join(OUT, "seeds"), exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(src, "*.csv"))):
        net = os.path.basename(p)[:-4]
        if net == "manifest":
            continue
        name = NAMES.get(net, net)
        dst = os.path.join(OUT, f"{name}.csv.gz")
        d = convert(p, dst)
        for s in range(5):
            convert(os.path.join(src, "seeds", f"{net}__s{s}.csv"), os.path.join(OUT, "seeds", f"{name}__s{s}.csv.gz"))
        m = man.loc[net]
        rows.append({"network": name, "file": os.path.basename(dst), "block_groups": len(d),
                     "block_groups_without_trips": int(d["F_1"].isna().sum()), "selected_seed": int(m.seed),
                     "L1_selected_seed": round(float(m.L1), 6), "sha256": sha256(dst)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print(len(rows), "networks;", sum(r["block_groups"] for r in rows), "block groups")


if __name__ == "__main__":
    main()
