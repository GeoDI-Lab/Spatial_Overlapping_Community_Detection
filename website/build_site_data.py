#!/usr/bin/env python
"""Build the data files of the results website (docs/) from the repository's data folder.

Reads
  data/results/                 per-city results (the paper's source data)
  data/memberships/             memberships F of the 50 cities and the Twin Cities metropolitan area
  website/assets/               US state outlines (Natural Earth 1:50m, public domain)
  $GUCD_CBG_SHAPES/<City>_cbg.shp   census block-group polygons of each city (TIGER/Line, USA Contiguous Albers),
                                    used only for drawing; set GUCD_CBG_SHAPES to the folder that holds them
Writes
  docs/data/cities.json         city-level results, national map, POI categories
  docs/data/city/<slug>.json    block-group polygons (simplified, quantized) and memberships of one city

  python website/build_site_data.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results")
MEMB = os.path.join(ROOT, "data", "memberships")
ASSETS = os.path.join(ROOT, "website", "assets")
OUT = os.path.join(ROOT, "docs", "data")
ALBERS = "ESRI:102003"                     # USA Contiguous Albers Equal Area Conic, as the block-group shapefiles

# comparators of Fig. 2b: column label in the source data -> (short key, family)
L1_ROWS = [
    ("Weighted overlap F (this study)", "F", "F"),
    ("Binary overlap", "binary", "own"), ("Hard partition", "partition", "own"),
    ("Weighted BigCLAM (BKN)", "wbigclam", "overlap"), ("BigCLAM", "bigclam", "overlap"),
    ("Link communities", "linkcom", "overlap"), ("Clique percolation (CPM)", "cpm", "overlap"),
    ("NOCD", "nocd", "overlap"), ("OCDDP", "ocddp", "overlap"),
    ("Infomap", "infomap", "partition"), ("Leiden", "leiden", "partition"), ("Louvain", "louvain", "partition"),
    ("node2vec (d = 10) + k-means", "n2v_km", "embedding"), ("node2vec (d = 10) vectors", "n2v_vec", "embedding"),
    ("MNE + k-means", "mne_km", "embedding"), ("MNE vectors (10 PCs)", "mne_vec", "embedding"),
    ("Degree only", "degree", "reference"), ("Constant", "constant", "reference"),
    ("Best of 12 plain conversions", "best12", "strongest"), ("Best of 57 conversions", "best57", "strongest"),
]
CPC_ROWS = [("Weighted overlap F (this study)", "F"), ("Pop+POI", "poppoi"), ("POI", "poi"),
            ("Population", "pop"), ("Radiation", "radiation"), ("Gravity (PL)", "grav_pl"),
            ("SC-gravity (PL)", "scgrav_pl"), ("Gravity (Exp)", "grav_exp"), ("SC-gravity (Exp)", "scgrav_exp")]
ZONE_VARS = {"a": "poi_density", "b": "poi_mix", "c": "res_density", "d": "dist_centre", "e": "trip_km"}
ALIAS = {"Louisville+Jefferson County metro government": "Louisville",       # network names used by several tables
         "Nashville-Davidson metropolitan government": "Nashville"}
SHAPE_NAME = {"Indianapolis": "Indianapolis city (balance)",                 # city -> shapefile stem, where they differ
              "Louisville": "Louisville+Jefferson County metro government (balance)",
              "Nashville": "Nashville-Davidson metropolitan government (balance)"}


def ix(path, col):
    """read a result table and index it by the city names of Supplementary Table 1"""
    d = pd.read_csv(path)
    d[col] = d[col].replace(ALIAS)
    return d.set_index(col)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def rd(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def flat_rings(geom, x0, y0, s):
    """polygon or multipolygon -> list of rings, each a flat list of integer coordinates in the city frame"""
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    out = []
    for p in polys:
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords)[:, :2]
            q = np.round(np.c_[(xy[:, 0] - x0) * s, (y0 - xy[:, 1]) * s]).astype(int)
            keep = np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]
            q = q[keep]
            if len(q) >= 4:
                out.append(q.ravel().tolist())
    return out


def national_map(cities):
    import geopandas as gpd
    from pyproj import Transformer
    st = gpd.read_file(os.path.join(ASSETS, "us_states_conus_ne50m_lakes.geojson")).to_crs(ALBERS)
    st["geometry"] = st.geometry.simplify(4000)
    minx, miny, maxx, maxy = st.total_bounds
    s = 1000.0 / (maxx - minx)
    rings = [r for g in st.geometry for r in flat_rings(g, minx, maxy, s)]
    tr = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    for c in cities:
        x, y = tr.transform(c["lon"], c["lat"])
        c["x"], c["y"] = round((x - minx) * s, 1), round((maxy - y) * s, 1)
    return {"w": 1000, "h": round((maxy - miny) * s), "states": rings}


def city_table():
    t = pd.read_csv(os.path.join(RES, "supplementary_tables", "Table_S_cities.csv"))
    l1 = ix(os.path.join(RES, "figure2", "source_data_Figure2_b.csv"), "city")
    red = ix(os.path.join(RES, "figure2", "source_data_Figure2_c.csv"), "city")
    cpc = ix(os.path.join(RES, "figure3", "source_data_Figure3_a.csv"), "city")
    allm = pd.read_csv(os.path.join(RES, "extended_data_table1", "per_city_all_metrics.csv"))
    allm["display"] = allm["display"].replace({"GUCD": "Weighted overlap F (this study)"})
    spear = allm.pivot(index="city_display", columns="display", values="s_corr")
    size = ix(os.path.join(RES, "extended_data_fig3", "source_data_ED3_a.csv"), "city_network")
    g = ix(os.path.join(RES, "figure5", "source_data_Figure5_b.csv"), "city")
    hinc = pd.read_csv(os.path.join(RES, "extended_data_fig5", "source_data_ED6_c.csv"))
    hinc = hinc[hinc.delta_raw.notna()].assign(city=lambda d: d.city.replace(ALIAS)).set_index("city")
    inc = ix(os.path.join(RES, "extended_data_fig6", "source_data_ED5_ab.csv"), "network")
    zones = {}
    for p, key in ZONE_VARS.items():
        z = pd.read_csv(os.path.join(RES, "extended_data_fig2", f"source_data_ED4_{p}.csv"))
        z = z[z.network_type.str.startswith("city")].assign(network=lambda d: d.network.replace(ALIAS))
        zones[key] = z.set_index("network")["effect"]
    cities = []
    for r in t.itertuples():
        c = r.city
        cities.append({
            "name": c, "slug": slug(c), "state": r.state,
            "lon": rd(inc.loc[c, "lon"], 5), "lat": rd(inc.loc[c, "lat"], 5),
            "pop": int(r.population_ACS2016_2020_network_block_groups), "n_cbg": int(r.block_groups_network),
            "trips": int(r.trips_between_block_groups),
            "L1": {k: rd(l1.loc[c, f"L1 | {lab}"]) for lab, k, _ in L1_ROWS},
            "red": {"binary": rd(red.loc[c, "pct_L1_reduction_by_F | Own binary overlap"], 2),
                    "partition": rd(red.loc[c, "pct_L1_reduction_by_F | Own hard partition"], 2),
                    "best12": rd(red.loc[c, "pct_L1_reduction_by_F | Best of 12 plain conversions"], 2),
                    "best57": rd(red.loc[c, "pct_L1_reduction_by_F | Best of 57 conversions, any kind"], 2)},
            "cpc": {k: rd(cpc.loc[c, lab]) for lab, k in CPC_ROWS},
            "spearman": {k: rd(spear.loc[c, lab]) for lab, k in CPC_ROWS},
            "overlap_mean": rd(size.loc[c, "mean_overlap_count_F_gt_0.5"], 3),
            "G_raw": rd(g.loc[c, "G_raw"], 3), "G_loc": rd(g.loc[c, "G_loc"], 3),
            "dHinc": rd(hinc.loc[c, "delta_raw"], 4), "dHinc_vol": rd(hinc.loc[c, "delta_vol"], 4),
            "rho_income": rd(inc.loc[c, "rho_paper_F"], 3), "income_level": inc.loc[c, "level"],
            "zone": {k: rd(v.loc[c], 3) for k, v in zones.items()},
        })
    return cities


def categories():
    a = pd.read_csv(os.path.join(RES, "figure5", "source_data_Figure5_a.csv"))
    ed = pd.read_csv(os.path.join(RES, "extended_data_fig4", "source_data_ED7.csv"))
    ed = ed[ed.fig5a_group.notna() & ed.shown_ge20_POIs].assign(city=lambda d: d.city.replace(ALIAS))
    per = {lab: {r.city: rd(r.log2_OR, 3) for r in sub.itertuples()} for lab, sub in ed.groupby("label")}
    return [{"label": r.label, "category": r.category, "group": "higher-order" if r.group == "D" else "everyday",
             "n": int(r.n_def), "n_gt1": int(r.n_L_gt1), "median": rd(r.median_log2OR, 3),
             "q25": rd(r.q25_log2OR, 3), "q75": rd(r.q75_log2OR, 3), "per_city": per.get(r.label, {})}
            for r in a.itertuples()]


def f1000(fm):
    """memberships x 1000 as integers; every F > 0.5 stays above 500 and every F > 0 above 0, so that overlap counts
    and membership are exact"""
    f = np.nan_to_num(fm)
    q = np.round(f * 1000).astype(int)
    q[(f > 0.5) & (q <= 500)] = 501
    q[(f > 0) & (q == 0)] = 1                            # tiny memberships stay nonzero
    return q


def city_maps(cities):
    """docs/data/city/<slug>.json: simplified block-group polygons in a local frame (longer side 2,000 units), rings as
    integer deltas, and the memberships F x 1000 (integers), rows in the order of data/memberships/<City>.csv.gz"""
    import geopandas as gpd
    shapes = os.environ.get("GUCD_CBG_SHAPES")
    if not shapes:
        raise SystemExit("set GUCD_CBG_SHAPES to the folder with <City>_cbg.shp (census block groups of each city)")
    for c in cities:
        F = pd.read_csv(os.path.join(MEMB, f"{c['name']}.csv.gz"), dtype={"GISJOIN": str})
        g = gpd.read_file(os.path.join(shapes, f"{c['shape']}_cbg.shp"))[["GISJOIN_1", "geometry"]]
        g = g.drop_duplicates("GISJOIN_1").set_index("GISJOIN_1").to_crs(ALBERS)
        miss = set(F.GISJOIN) - set(g.index)
        if miss:
            raise SystemExit(f"{c['name']}: {len(miss)} block groups without a polygon")
        geo = g.loc[F.GISJOIN, "geometry"]
        minx, miny, maxx, maxy = geo.total_bounds
        s = 2000.0 / max(maxx - minx, maxy - miny)
        geo = geo.simplify(max(12.0, max(maxx - minx, maxy - miny) / 2500), preserve_topology=True)
        rings, cx, cy = [], [], []
        for geom in geo:
            rs = flat_rings(geom, minx, maxy, s)
            rings.append([[r[0], r[1]] + np.diff(np.asarray(r).reshape(-1, 2), axis=0).ravel().tolist() for r in rs])
            p = geom.representative_point()
            cx.append(round((p.x - minx) * s, 1)); cy.append(round((maxy - p.y) * s, 1))
        fcols = [f"F_{k}" for k in range(1, 11)]
        fm = F[fcols].to_numpy()
        iso = np.flatnonzero(np.isnan(fm).any(axis=1))        # block groups without trips to others: no memberships
        doc = {"name": c["name"], "state": c["state"], "K": 10, "w": round((maxx - minx) * s), "h": round((maxy - miny) * s),
               "ids": F.GISJOIN.tolist(), "F": f1000(fm).tolist(),
               "iso": iso.tolist(), "rings": rings, "cx": cx, "cy": cy}
        path = os.path.join(OUT, "city", f"{c['slug']}.json")
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        print(f"  {c['slug']}.json: {len(F)} block groups, {os.path.getsize(path) / 1e3:.0f} kB")


def page():
    """docs/index.html: website/page.html (head part, then the body after '<!-- body -->') in a full HTML document"""
    src = open(os.path.join(ROOT, "website", "page.html"), encoding="utf-8").read()
    head, body = src.split("<!-- body -->", 1)
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
    with open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    open(os.path.join(ROOT, "docs", ".nojekyll"), "w").close()        # serve the folder as is on GitHub Pages


def main():
    os.makedirs(os.path.join(OUT, "city"), exist_ok=True)
    page()
    cities = city_table()
    for c in cities:                                      # shapefile names use the network names of Supplementary Table 1
        c["shape"] = SHAPE_NAME.get(c["name"], c["name"])
    if os.path.isdir(MEMB):
        city_maps(cities)
    for c in cities:
        c.pop("shape")
    nat = national_map(cities)
    meta = {"methods": [{"label": lab, "key": k, "family": fam} for lab, k, fam in L1_ROWS],
            "generation_models": [{"label": lab, "key": k} for lab, k in CPC_ROWS]}
    doc = {"meta": meta, "map": nat, "cities": cities, "categories": categories()}
    with open(os.path.join(OUT, "cities.json"), "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"cities.json: {len(cities)} cities, {os.path.getsize(os.path.join(OUT, 'cities.json')) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
