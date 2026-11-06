# Memberships F

`<network>.csv.gz`: the memberships used in the paper for the 50 cities and the Twin Cities metropolitan area
(weekday and weekend networks), one row per census block group in the order of the network: `GISJOIN` (NHGIS
identifier), `lat` and `lon` (internal point), and `F_1` ... `F_10` (community k of the paper is column `F_k`).
Rows of block groups without trips to other block groups of their network are empty. `seeds/` holds the fits of the
five training seeds (0-4); the selected seed has the lowest L1 on the observed pairs. `manifest.csv` lists, for each
network, the number of block groups, those without trips, the selected seed, its L1 and a SHA-256 checksum.
