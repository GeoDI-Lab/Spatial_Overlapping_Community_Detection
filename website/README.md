# Atlas

`page.html` is the page; `build_site_data.py` writes `docs/index.html` and the data in `docs/data/` from
`data/results/`, `data/memberships/` and the census block-group polygons of each city (TIGER/Line; set
`GUCD_CBG_SHAPES` to the folder with `<City>_cbg.shp`). `docs/` is served as is (GitHub Pages).

`build_poi_layer.py` writes `docs/data/poi/`, the layer of the map of points of interest and flows: the number of
points of interest of each block group by order and type (counts only; the page places them at random inside their
block group), the strongest flows implied by the memberships between communities, and the central-place contrast by
overlap count. It needs the SafeGraph points of interest joined to the block groups, which this repository does not
hold (set `GUCD_POI_CACHE`), and runs after `build_site_data.py`.
