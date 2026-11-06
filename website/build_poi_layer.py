"""Atlas layer "Points of interest and flows": one JSON per city in docs/data/poi/.

  - POIs per block group, by order (higher-order and everyday: the 30 SafeGraph categories assigned before the
    analysis, Supplementary Table 11; other: all remaining POIs) and by SafeGraph group (11 groups and none). Only
    counts per block group are written; the atlas draws them as dots at random positions inside each block group.
  - The strongest flows implied by the memberships between block groups whose largest memberships lie in different
    communities: the 250 pairs with the largest F_u.F_v (expected trips exp(F_u.F_v)), at most 10 per block group
    (a display rule: without it, a few block groups take most of the pairs).
  - The central-place contrast G of the city by overlap count (Supplementary Table 12) and the medians over cities.

The POIs come from SafeGraph Places joined to the block groups, which this repository does not hold: set
GUCD_POI_CACHE to the folder with <City>.parquet (columns top_category, cls, GISJOIN). Block-group order and F are
read from docs/data/city/<slug>.json, so run build_site_data.py first.

  python website/build_poi_layer.py
"""
import os
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs", "data")
TABLES = os.path.join(ROOT, "data", "results", "supplementary_tables")
CACHE = os.environ.get("GUCD_POI_CACHE", "")
GROUPS = ["Retail", "Dining", "Entertainment", "Travel", "Educational", "Financial", "Business", "Automotive",
          "Real", "Transportation", "Industrial", ""]                  # SafeGraph groups as in Fig. 4; "" = none
ORDERS = ["higher", "everyday", "other"]
STEM = {"Louisville": "Louisville+Jefferson_County_metro_government",
        "Nashville": "Nashville-Davidson_metropolitan_government"}
N_FLOWS, CAP, PER_ROW = 250, 10, 30


def categories():
    t = pd.read_csv(os.path.join(TABLES, "Table_S_categories.csv"))
    return {r.safegraph_category: (0 if r.group == "Higher-order" else 1) for r in t.itertuples()}


def graded():
    per = pd.read_csv(os.path.join(TABLES, "Table_S_gradedG_per_city.csv"))
    per["city"] = per.city.replace({"Louisville+Jefferson County metro government": "Louisville",
                                    "Nashville-Davidson metropolitan government": "Nashville"})
    nat = pd.read_csv(os.path.join(TABLES, "Table_S_gradedG.csv"))
    nat = nat[nat.part == "a"].set_index("level")
    med = {k: round(float(nat.loc[k, "median_G"]), 3) for k in ("2", "3", "ge4")}
    return per.set_index("city"), med


def top_flows(F):
    """the N_FLOWS pairs u < v with the largest F_u.F_v among block groups with different largest memberships,
    taken in descending order with at most CAP pairs per block group (candidates: the PER_ROW best of each row)"""
    alive = F.sum(1) > 0
    main = np.where(alive, F.argmax(1), -1)
    cand, n = [], len(F)
    for i0 in range(0, n, 512):
        G = F[i0:i0 + 512] @ F.T
        for r in range(G.shape[0]):
            i = i0 + r
            if not alive[i]:
                continue
            row = G[r]
            js = np.where((np.arange(n) > i) & alive & (main != main[i]) & (row > 0))[0]
            if not len(js):
                continue
            for j in js[np.argsort(-row[js])[:PER_ROW]]:
                cand.append((float(row[j]), i, int(j)))
    out, deg = [], {}
    for v, i, j in sorted(cand, reverse=True):
        if deg.get(i, 0) < CAP and deg.get(j, 0) < CAP:
            out.append([i, j, round(v, 3)])
            deg[i], deg[j] = deg.get(i, 0) + 1, deg.get(j, 0) + 1
            if len(out) == N_FLOWS:
                break
    return out


def main():
    if not os.path.isdir(CACHE):
        raise SystemExit("set GUCD_POI_CACHE to the folder with the POI tables (<City>.parquet)")
    cats, (per, med) = categories(), graded()
    os.makedirs(os.path.join(DOCS, "poi"), exist_ok=True)
    cities = json.load(open(os.path.join(DOCS, "cities.json")))["cities"]
    total = 0
    for c in cities:
        doc = json.load(open(os.path.join(DOCS, "city", c["slug"] + ".json")))
        ids = {g: i for i, g in enumerate(doc["ids"])}
        F = np.asarray(doc["F"], float) / 1000.0
        poi = pd.read_parquet(os.path.join(CACHE, STEM.get(c["name"], c["name"].replace(" ", "_")) + ".parquet"))
        poi = poi[poi.GISJOIN.isin(ids)]
        order = poi.top_category.map(cats).fillna(2).astype(int).to_numpy()
        group = poi.cls.fillna("").map({g: k for k, g in enumerate(GROUPS)}).fillna(len(GROUPS) - 1).astype(int)
        code = group.to_numpy() * 3 + order                  # one code per POI: group x order
        node = poi.GISJOIN.map(ids).to_numpy()
        counts = pd.Series(1, index=pd.MultiIndex.from_arrays([node, code])).groupby(level=[0, 1]).sum()
        cbg = [[] for _ in ids]
        for (i, k), v in counts.items():
            cbg[i] += [int(k), int(v)]
        g = per.loc[c["name"]] if c["name"] in per.index else None
        out = {"orders": ORDERS, "groups": [x or "None" for x in GROUPS], "cbg": cbg, "flows": top_flows(F),
               "G": None if g is None else {"2": round(float(g.G_2), 3), "3": round(float(g.G_3), 3),
                                             "ge4": round(float(g.G_ge4), 3)},
               "G_median": med, "n": {o: int((order == k).sum()) for k, o in enumerate(ORDERS)}}
        with open(os.path.join(DOCS, "poi", c["slug"] + ".json"), "w") as f:
            json.dump(out, f, separators=(",", ":"))
        total += len(poi)
        print(f"{c['name']}: {len(poi):,} POIs ({out['n']['higher']:,} higher-order, {out['n']['everyday']:,} "
              f"everyday), {len(out['flows'])} flows")
    print(f"{total:,} POIs in {len(cities)} cities")


if __name__ == "__main__":
    main()
