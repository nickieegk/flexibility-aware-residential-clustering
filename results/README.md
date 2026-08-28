# Curated result tables

This directory contains compact tabular outputs that support the main quantitative results reported in the associated manuscript. They are included here to make the principal comparisons inspectable without rerunning the full DTW workflow.

Included files:

- `3_clustering_evaluation_results.csv` — DTW k-sweep metrics (`k=2,...,20`), including inertia, silhouette coefficient and average peak-match score.
- `6_delta_load_clusters_comparison.csv` — cluster-level descriptive statistics for DTW load-based and flexibility-aware clustering at `k=8`.
- `7_inter_cluster_variance.csv` — cluster-level mean `delta_load`, standard errors, 95% confidence intervals and inter-cluster variance.
- `8_dr_targeting_scenarios.csv` — principal DTW versus flexibility-aware DR-targeting scenarios at `k=8`.
- `9_kruskal_wallis.csv` — Kruskal–Wallis tests for `delta_load`, SEF, PSS and PAR under DTW and flexibility-aware clustering.
- `8_DTW_vs_Euklidian_dr_targeting_scenarios.csv` — DTW versus Euclidean targeting scenarios.
- `8_DTW_vs_Euklidian_kruskal_wallis.csv` — DTW versus Euclidean Kruskal–Wallis results.
- `sensitivity_targeting_summary.csv` — compact targeting summary for `k=4,...,12`, including participation, captured DR potential and targeting efficiency.

## Important note on the `delta_load` scale

The original implementation calculates variables named `*_energy` by summing 15-minute interval mean-power values. Consequently, the stored absolute `delta_load` values are in the original analysis scale rather than physically converted Wh. For a 15-minute interval, conversion of these absolute summed-power quantities to Wh requires multiplication by `0.25 h`.

This common scaling does **not** change SEF or PSS ratios, cluster assignments based on the stated features, cluster rankings by `delta_load`, Kruskal–Wallis/Dunn results, fractions of DR potential captured, or targeting-efficiency ratios. Any manuscript figure or table that reports an absolute energy quantity should apply the same conversion consistently.

## Full underlying and extended data

The complete derived household-day tables, pairwise Dunn outputs, full `k=4,...,12` results and publication/extended-data figures are intended for a separate archival research-data deposit under CC BY 4.0. The original PLEGMA source data are not redistributed here.
