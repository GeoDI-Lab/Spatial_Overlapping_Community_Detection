# Per-city results (source data)

One folder per figure or table of the paper. The file names keep the names of the scripts that wrote them, so the
Extended Data files carry the working numbers of the figures:

| Folder | Paper item | Files |
|---|---|---|
| `figure1/` | Fig. 1, the 50 cities (b), flow by decile of shared membership (c), trips linking two communities (d: all 1,412 pairs of communities and the five maps) | `source_data_Figure1_b/c/d_pairs/d_maps.csv` |
| `figure2/` | Fig. 2, flow reconstruction on observed pairs | `source_data_Figure2_a/b/b_summary/c.csv` |
| `figure3/` | Fig. 3, flow generation for held-out origins | `source_data_Figure3_a/b_CPC/b_Spearman/b_summary.csv` |
| `figure4/` | Fig. 4, communities of the Twin Cities metropolitan area (a–c) and Austin (d–f); overlap zone vs rest in 50 cities (g) | `source_data_Figure4_*.csv` |
| `figure5/` | Fig. 5, central-place categories, contrast G, correlations | `source_data_Figure5_a/b/c.csv` |
| `extended_data_fig1/` | Extended Data Fig. 1, number of communities and binarization threshold | `source_data_ED2_a/b.csv` |
| `extended_data_fig2/` | Extended Data Fig. 2, place characteristics of the overlap zone | `source_data_ED4_*.csv` |
| `extended_data_fig3/` | Extended Data Fig. 3, city population and overlap count | `source_data_ED3_*.csv` |
| `extended_data_fig4/` | Extended Data Fig. 4, odds ratios of 62 categories in 50 cities | `source_data_ED7.csv` |
| `extended_data_fig5/` | Extended Data Fig. 5, socio-economic associations | `source_data_ED6_*.csv` |
| `extended_data_fig6/` | Extended Data Fig. 6, overlap and residents' income | `source_data_ED5_*.csv` |
| `extended_data_table1/` | Extended Data Table 1 and the per-city values of all models and metrics | `main_table.csv`, `per_city_all_metrics.csv` (the model `GUCD` is F), `wins_wilcoxon_holm.csv` |
| `supplementary_fig1/` | Supplementary Fig. 1, mean daily visits per device by city (the distributions come from device-level records, which are not shared) | `source_data_FigureS_visits_per_device.csv` |
| `supplementary_fig2/` | Supplementary Fig. 2, spectral input features | `source_data_FigureS_robustness_spectral_input.csv` |
| `supplementary_fig3/` | Supplementary Fig. 3, overlap count in Chicago, New York and Los Angeles | `source_data_FigureS_landmarks_*.csv` |
| `supplementary_tables/` | Supplementary Tables 1–12 | `Table_S_cities.csv` (1), `Table_S_task1_methods.csv` (2), `Table_S_task1_all_conversions.csv` (3), `Table_S_task1_per_city.csv` (4), `Table_S_task1_seeds_node2vec.csv` (5), `Table_S_spectral_comparison.csv` (6), `Table_S_spectral_thresholds.csv` (7), `Table_S_task2_medians.csv` (8), `Table_S_task2_per_city_CPC.csv` (9), `Table_S_overlap_by_city.csv` (10), `Table_S_categories.csv` (11), `Table_S_gradedG.csv` and `Table_S_gradedG_per_city.csv` (12) |

Louisville and Nashville appear in several files under the names of their consolidated governments
("Louisville+Jefferson County metro government", "Nashville-Davidson metropolitan government").
