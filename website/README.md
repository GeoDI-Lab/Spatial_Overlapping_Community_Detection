# Atlas

`page.html` is the page; `build_site_data.py` writes `docs/index.html` and the data in `docs/data/` from
`data/results/`, `data/memberships/` and the census block-group polygons of each city (TIGER/Line; set
`GUCD_CBG_SHAPES` to the folder with `<City>_cbg.shp`). `docs/` is served as is (GitHub Pages).
