# %% [markdown]
# # 9. Clusterings για Διαφορετικές Τιμές k
#
# Εκτελεί ξανά τα δύο βασικά clusterings για k=4..12, εκτός από k=8 που έχει
# ήδη υπολογιστεί στα προηγούμενα notebooks.
#
# - Load-based clustering: DTW time-series k-means πάνω στα 96-σημείων ημερήσια
#   προφίλ κατανάλωσης.
# - Flexibility-aware clustering: k-means πάνω στα metrics `SEF`, `PSS`, `PAR`
#   και `Slack_i`.
#
# Δεν υπολογίζεται precomputed DTW distance matrix εδώ, επειδή δεν χρειάζεται
# silhouette/evaluation sweep. Κρατάμε τα δεδομένα φορτωμένα/κανονικοποιημένα μία
# φορά και τρέχουμε μόνο τα τελικά clusterings για κάθε k.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config

import multiprocessing
multiprocessing.freeze_support()

import tempfile

import pandas as pd

from helpers import (
    load_daily_profiles,
    load_flexibility_metrics,
    normalise_daily_profiles,
    run_dr_targeting_analysis,
    run_dtw_profile_clustering_parallel,
    run_flexibility_kmeans_clustering,
    run_kruskal_dunn_analysis,
    scale_flexibility_features,
)


K_VALUES = [k for k in range(4, 13)]

# Keep this False when you only want to regenerate comparison graphs/statistics
# from the already-created OUTPUT_CSV.
RUN_CLUSTERINGS = False

RANDOM_STATE = 42

SAKOE_CHIBA_RADIUS = 6

LOAD_MAX_ITER_FINAL = 15
LOAD_N_INIT_FINAL   = 5

FLEX_MAX_ITER_FINAL = 300
FLEX_N_INIT_FINAL   = 5

N_CPU          = os.cpu_count() or 1
LOAD_N_WORKERS = min(N_CPU, LOAD_N_INIT_FINAL)

FEATURE_COLS = ["SEF", "PSS", "PAR", "Slack_i"]
STAT_TEST_METRICS = ["delta_load", "SEF", "PSS", "PAR"]
META_COLS    = ["house_id", "date"]

LOAD_INPUT_CSV = config.DATA_PATH + "/2_targeted_period_imputated_daily_profiles.csv"
FLEX_INPUT_CSV = config.OUTPUT_PATH + "/4_days_metrics.csv"
OUTPUT_CSV     = config.DATA_PATH + "/9_different_ks_CLUSTERED_both_ways.csv"


def add_load_cluster_column(
    df_metrics: pd.DataFrame,
    load_meta: pd.DataFrame,
    labels_0,
    cluster_col: str,
) -> pd.DataFrame:
    df_labels = load_meta.copy()
    df_labels[cluster_col] = labels_0 + 1

    duplicate_keys = df_labels.duplicated(META_COLS).sum()
    assert duplicate_keys == 0, (
        f"Βρέθηκαν {duplicate_keys} διπλά house_id/date στο load profile input."
    )

    merged = df_metrics.merge(df_labels, on=META_COLS, how="left", validate="one_to_one")
    missing_labels = merged[cluster_col].isna().sum()
    assert missing_labels == 0, (
        f"Δεν βρέθηκε load-based cluster label για {missing_labels} γραμμές metrics."
    )

    merged[cluster_col] = merged[cluster_col].astype(int)
    return merged


def expected_cluster_columns(k_values: list[int]) -> list[str]:
    cols = []
    for k in k_values:
        cols.extend([f"k_{k}_lb_cluster_id", f"k_{k}_fa_cluster_id"])
    return cols


print("Φόρτωση δεδομένων για διαφορετικές τιμές k.")
print(f"k values: {K_VALUES}")

expected_cols = expected_cluster_columns(K_VALUES)

if RUN_CLUSTERINGS:
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        missing_cols = [c for c in expected_cols if c not in df_existing.columns]
    else:
        missing_cols = expected_cols

    if not missing_cols:
        print(f"Το clustered αρχείο υπάρχει ήδη με όλες τις στήλες: '{OUTPUT_CSV}'")
        print("Δεν γίνεται επανυπολογισμός cluster labels.")
    else:
        print(f"Θα υπολογιστούν cluster labels. Λείπουν στήλες: {missing_cols}")

        X_load_raw, load_meta, _df_load_raw, _interval_cols = load_daily_profiles(LOAD_INPUT_CSV)
        X_load_scaled = normalise_daily_profiles(X_load_raw)

        X_flex_raw, df_out = load_flexibility_metrics(
            FLEX_INPUT_CSV,
            feature_cols=FEATURE_COLS,
            meta_cols=META_COLS,
        )
        X_flex_scaled = scale_flexibility_features(X_flex_raw)

        with tempfile.TemporaryDirectory(prefix="different_k_load_clusterings_") as mmap_dir:
            for k in K_VALUES:
                print(f"\nLoad-based DTW clustering | k={k}")
                load_labels_0, _load_centroids, load_inertia = run_dtw_profile_clustering_parallel(
                    X_scaled     = X_load_scaled,
                    mmap_dir     = mmap_dir,
                    k            = k,
                    radius       = SAKOE_CHIBA_RADIUS,
                    max_iter     = LOAD_MAX_ITER_FINAL,
                    n_init       = LOAD_N_INIT_FINAL,
                    random_state = RANDOM_STATE,
                    n_workers    = LOAD_N_WORKERS,
                )

                load_cluster_col = f"k_{k}_lb_cluster_id"
                df_out = add_load_cluster_column(
                    df_metrics  = df_out,
                    load_meta   = load_meta,
                    labels_0    = load_labels_0,
                    cluster_col = load_cluster_col,
                )
                print(f"{load_cluster_col}: inertia={load_inertia:.4f}")

                print(f"Flexibility-aware k-means clustering | k={k}")
                flex_labels_0, _flex_model = run_flexibility_kmeans_clustering(
                    X_scaled     = X_flex_scaled,
                    k            = k,
                    max_iter     = FLEX_MAX_ITER_FINAL,
                    n_init       = FLEX_N_INIT_FINAL,
                    random_state = RANDOM_STATE,
                )

                flex_cluster_col = f"k_{k}_fa_cluster_id"
                df_out[flex_cluster_col] = flex_labels_0 + 1

                print(f"{flex_cluster_col}:")
                print(df_out[flex_cluster_col].value_counts().sort_index().to_string())

        df_out.to_csv(OUTPUT_CSV, index=False)

        print(f"\nΑποτελέσματα → '{OUTPUT_CSV}'")
        print(f"Προστέθηκαν {2 * len(K_VALUES)} στήλες cluster labels.")
else:
    assert os.path.exists(OUTPUT_CSV), (
        f"Δεν υπάρχει το '{OUTPUT_CSV}'. Θέσε RUN_CLUSTERINGS=True για να δημιουργηθεί."
    )
    print("RUN_CLUSTERINGS=False: παράλειψη υπολογισμού cluster labels.")

print(f"\nΦόρτωση clustered labels από αρχείο: '{OUTPUT_CSV}'")
df_for_plots = pd.read_csv(OUTPUT_CSV)
missing_cols = [c for c in expected_cols if c not in df_for_plots.columns]
assert not missing_cols, (
    f"Το '{OUTPUT_CSV}' δεν περιέχει όλες τις αναμενόμενες στήλες: {missing_cols}"
)

print("\nΔημιουργία comparison graphs για κάθε k.")
for k in K_VALUES:
    load_cluster_col = f"k_{k}_lb_cluster_id"
    flex_cluster_col = f"k_{k}_fa_cluster_id"

    print(f"\nDR targeting comparison | k={k}")
    scenarios_df = run_dr_targeting_analysis(
        df                       = df_for_plots,
        load_cluster_col         = load_cluster_col,
        flexibility_cluster_col  = flex_cluster_col,
        output_csv               = config.DATA_PATH + f"/9_k_{k}_dr_targeting_scenarios.csv",
        output_png               = config.DATA_PATH + f"/9_k_{k}_dr_targeting_scenarios.png",
        title_suffix             = f"k={k}",
    )
    print(
        scenarios_df[
            ["method", "scenario", "selected_days", "total_days", "pct_days_selected", "pct_dr_captured"]
        ].to_string(index=False)
    )

    print(f"\nKruskal-Wallis / Dunn comparison | k={k}")
    kw_df, sig_pairs_df = run_kruskal_dunn_analysis(
        df                       = df_for_plots,
        load_cluster_col         = load_cluster_col,
        flexibility_cluster_col  = flex_cluster_col,
        output_kw_csv            = config.DATA_PATH + f"/9_k_{k}_kruskal_wallis.csv",
        output_sig_pairs_csv     = config.DATA_PATH + f"/9_k_{k}_dunn_significant_pairs.csv",
        output_kw_png            = config.DATA_PATH + f"/9_k_{k}_kruskal_wallis.png",
        dunn_heatmap_prefix      = config.DATA_PATH + f"/9_k_{k}_dunn_heatmaps",
        metrics                  = STAT_TEST_METRICS,
        title_suffix             = f"k={k}",
    )
    print("=== Kruskal-Wallis Results ===")
    print(kw_df.to_string(index=False))
    print("\n=== Statistically Significant Dunn Pairs (p < 0.05, Holm correction) ===")
    if sig_pairs_df.empty:
        print("No significant pairs found.")
    else:
        print(sig_pairs_df.to_string(index=False))
# %%
