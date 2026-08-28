# Flexibility-Aware Clustering of Residential Electricity Consumption for Demand Response Targeting Using High-Resolution Smart Meter Data

This repository contains the reproducible analysis workflow associated with the manuscript **“Flexibility-Aware Clustering of Residential Electricity Consumption for Demand Response Targeting Using High-Resolution Smart Meter Data.”**

## Authors

- **Nickie Gkolia** — Decision Support Systems Laboratory, School of Electrical and Computer Engineering, National Technical University of Athens, Athens, Greece — `ngkolia@epu.ntua.gr`
- **Nikos Dimitropoulos** — Decision Support Systems Laboratory, School of Electrical and Computer Engineering, National Technical University of Athens, Athens, Greece — `ndimitropoulos@epu.ntua.gr`
- **Vangelis Marinakis** — Decision Support Systems Laboratory, School of Electrical and Computer Engineering, National Technical University of Athens, Athens, Greece — `vmarinakis@epu.ntua.gr`

## Study overview

The study develops and evaluates a flexibility-aware framework for residential electricity-consumption segmentation for demand-response targeting. Three clustering approaches are compared:

1. **DTW load-profile clustering** of standardized 15-minute daily load profiles.
2. **Euclidean load-profile clustering** of the same standardized daily profiles.
3. **Flexibility-aware clustering** using shiftable-energy fraction (SEF), peak shiftability share (PSS), peak-to-average ratio (PAR), and household flexibility slack.

The resulting segments are evaluated using descriptive statistics, Kruskal–Wallis tests, Dunn post-hoc comparisons, and downstream demand-response targeting experiments.

## Source dataset

The analysis uses the publicly available **PLEGMA Dataset**. The original household data are not redistributed in this repository.

- **Dataset:** PLEGMA Dataset, University of Strathclyde
- **Dataset DOI:** `10.15129/3b01a6c6-2efd-424a-b8b8-5fe7fa445ded`
- **Dataset license:** CC BY 4.0
- **Dataset article:** Athanasoulias et al. (2024), *Scientific Data*, 11, 376
- **Article DOI:** `10.1038/s41597-024-03208-0`

Users wishing to reproduce the analysis should obtain the source data from the official PLEGMA repository and preserve the original attribution and license information.

## Analysis workflow

The numbered Python scripts form a sequential pipeline:

1. `1_data_reading.py` — reads and merges household electricity and environmental data and aggregates electricity measurements to 15-minute intervals.
2. `2_imputation.py` — restricts the study period, treats missing values, and creates complete household-day profiles.
3. `3_daily_profiles_load_clustering.py` — performs the DTW clustering sweep and the principal DTW daily-profile clustering.
4. `4_metrics.py` — calculates SEF, PSS, PAR, appliance start-time flexibility slack, and the demand-response-oriented evaluation quantity.
5. `5_flexibility_aware_clustering.py` — performs k-means clustering in the standardized flexibility-feature space.
6. `6_clustering_comparison.py` — compares DTW load-based and flexibility-aware clustering using descriptive statistics, statistical tests, and DR targeting scenarios.
7. `7_euklidean_load_clustering.py` — generates the Euclidean load-profile baseline.
8. `8_DTW_vs_Euklidian.py` — compares DTW and Euclidean load-profile clustering.
9. `9_different_k_values_clusterings.py` — evaluates robustness across `k = 4, ..., 12`.

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── config.py
├── helpers.py
├── source_code/
├── Data/
└── Outputs/
```

`Data/House_*` source folders are intentionally excluded from version control. Derived datasets and extended-data materials supporting the manuscript will be archived separately with a persistent DOI before submission.

## Reproducing the analysis

1. Download the original PLEGMA Dataset from the official repository.
2. Place the required `House_*` folders under `Data/`.
3. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

4. From the `source_code/` directory, run the scripts sequentially from Step 1 to Step 9.

The principal analyses use 15-minute daily profiles (96 observations per day), a fixed random state of 42 where applicable, and a Sakoe–Chiba radius of six 15-minute intervals for DTW clustering.

## Availability

The original PLEGMA source data are available through the DOI above. This GitHub repository provides the analysis code. A versioned archival release and the derived/extended data supporting the article will be deposited in a persistent research-data repository before manuscript submission; the final DOI will be added here and to the manuscript.
