# Mobility flows reveal an overlapping mesoscale in 50 US cities

Code, data and interactive results for the manuscript of the same title (under review), by Peng Luo, Di Zhu,
Meicheng Xiong, Sheng Wang, Fábio Duarte, Qi Ryan Wang, Paolo Santi and Carlo Ratti.

**Interactive atlas:** <https://peng-luo.com/mesoscale/>

## The model

Places are census block groups, and the mobility network of a city counts the trips between them (A = OD + ODᵀ,
five consecutive weekdays, 19–23 July 2021). Each block group u has a weighted membership F<sub>uk</sub> ≥ 0 in each
of K = 10 communities, and the log flow between two block groups is approximated by the inner product of their
memberships:

    log A_uv ≈ F_u · F_v = Σ_k F_uk F_vk

A two-layer graph convolutional network with a node2vec embedding of the network as input infers F from the flows by
minimizing the mean absolute error of the log flows on the observed pairs (L1). Each network is fitted with five
training seeds and F is the fit with the lowest L1. A block group's overlap count is the number of communities in
which its membership exceeds 0.5; the overlap zone of a city consists of the block groups with an overlap count of at
least 2.

## Repository

| Path | Content |
|---|---|
| `gucd/` | The package: networks and observed pairs (`data`), the node2vec input (`node2vec`), the GCN and its training (`model`), F, its binary version and its hard partition as fitted flow models (`conversions`), overlap count and overlap zone (`overlap`), and the evaluation of generated flows (`metrics`) |
| `scripts/fit_memberships.py` | Infers F for one network as in the paper (node2vec, seeds 0–4, best L1) |
| `demo/` | Demo on New York; its network is in `demo/data/` |
| `data/memberships/` | The memberships F used in the paper: 50 cities and the Twin Cities metropolitan area (weekday and weekend), selected seed and all five seeds |
| `data/results/` | The per-city results behind every figure and table of the paper (source data) |
| `docs/` | The interactive atlas (served with GitHub Pages); `website/` builds its data |
| `tools/` | How `data/memberships` was exported |

## Installation

Python 3.10 or later:

```bash
pip install -r requirements.txt
```

The package needs numpy, scipy, pandas and PyTorch (CPU is enough). Building the atlas data also needs geopandas,
shapely and pyproj.

## Quick start

```bash
python demo/run_demo.py
```

The demo loads New York's network and the published memberships and reproduces New York's reconstruction errors of
Fig. 2a: L1 = 0.605049 for F, 0.643424 for its binary version and 0.671910 for its hard partition. With `--refit`
it also infers F anew with one training seed (about 15–40 minutes on a laptop CPU for New York, the largest of the 50
networks).

To infer F for your own network (a symmetric matrix of trips between places, saved with `scipy.sparse.save_npz`):

```bash
python scripts/fit_memberships.py --network A_City.npz --out F_City.csv.gz
```

## Reproducibility

With two threads and deterministic algorithms, the package reproduces the paper's fits bit for bit: for Raleigh
(seed 0), the node2vec embedding and F are identical to those of the paper (maximum absolute difference 0) and the fit
stops at the same step (PyTorch 2.13 on CPU). On other platforms or library versions, fits can differ in the last
digits.

## Data

- `data/memberships/<network>.csv.gz`: one row per census block group in the order of the network, with its GISJOIN
  identifier, the latitude and longitude of its internal point, and the memberships `F_1` … `F_10` (community k of the
  paper is column `F_k`). Rows of block groups without trips to other block groups of their network are empty.
  `seeds/` holds the five training seeds; `manifest.csv` lists the selected seed, its L1 and a SHA-256 checksum.
- `data/results/`: the values plotted or tabulated in the paper, in one folder per figure or table (see the
  README there).
- `demo/data/A_New York.npz`: New York's network (trips between block groups, aggregated).

The mobility data are anonymized device records from PlaceIQ (now Precisely); they are commercial and available from
the provider under its terms. Points of interest are from SafeGraph, available under licence. Residents' income and
race are American Community Survey 2016–2020 five-year estimates from IPUMS NHGIS (https://www.nhgis.org).
Block-group boundaries are from the US Census Bureau (TIGER/Line).

## Citation

Luo, P., Zhu, D., Xiong, M., Wang, S., Duarte, F., Wang, Q. R., Santi, P. & Ratti, C. Mobility flows reveal an
overlapping mesoscale in 50 US cities. Manuscript under review (2026).
