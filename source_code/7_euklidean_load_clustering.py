# %% [markdown]
# # 7. Euclidean Baseline Ομαδοποίησης Προφίλ Κατανάλωσης
#
# Αυτό το notebook διαβάζει το
# `2_targeted_period_imputated_daily_profiles.csv`, τον ίδιο πίνακα
# κατοικίας-ημέρας που χρησιμοποιείται στο DTW load clustering, με 96 τιμές
# συνολικής κατανάλωσης 15λεπτου ανά ημέρα.
#
# Αντιμετωπίζει κάθε ημερήσιο προφίλ ως fixed-length διάνυσμα, κανονικοποιεί
# τις 96 στήλες χρονικών διαστημάτων και εκτελεί παράλληλες αρχικοποιήσεις
# k-means με απλή Euclidean distance. Η καλύτερη εκτέλεση επιλέγεται με βάση το
# inertia.
#
# Το αποτέλεσμα είναι το `7_days_metrics_CLUSTERED_three_ways.csv`, δηλαδή ο
# πίνακας metrics από το προηγούμενο βήμα εμπλουτισμένος με
# `euklidean_cluster_id`. Αυτό δίνει ένα απλούστερο baseline για σύγκριση με τα
# DTW-based ημερήσια load clusters.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config

import warnings
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.cluster import KMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

warnings.filterwarnings("ignore")

K            = 8
N_INIT       = 5
MAX_ITER     = 300
RANDOM_STATE = 42
N_CPU        = os.cpu_count() or 1
N_WORKERS    = min(N_CPU, N_INIT)

INPUT_CSV             = config.DATA_PATH + "/2_targeted_period_imputated_daily_profiles.csv"
CLUSTERED_METRICS_CSV = config.DATA_PATH + "/5_days_metrics_CLUSTERED_both_ways.csv"
OUTPUT_CSV            = config.DATA_PATH + "/7_days_metrics_CLUSTERED_three_ways.csv"


def load_profiles(path):
    df_raw = pd.read_csv(path)

    meta_cols = ["house_id", "date"]
    assert all(c in df_raw.columns for c in meta_cols), (
        "Λείπουν οι στήλες 'house_id' ή/και 'date'."
    )

    interval_cols = [c for c in df_raw.columns if c not in meta_cols]

    assert len(interval_cols) == 96, (
        f"Αναμένονταν 96 χρονικά διαστήματα, βρέθηκαν {len(interval_cols)}."
    )

    X_raw = df_raw[interval_cols].values.astype(np.float64)

    print(f"Loaded: {X_raw.shape[0]} profiles × {X_raw.shape[1]} intervals.")
    return X_raw, df_raw


def normalise_profiles(X_raw):
    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)

    X_scaled_3d = scaler.fit_transform(X_raw[:, :, np.newaxis])

    # Το sklearn KMeans θέλει 2D input: (n_samples, n_features)
    X_scaled_2d = X_scaled_3d[:, :, 0]

    return X_scaled_2d


def _single_init_worker(mmap_path, k, max_iter, seed):
    X = load(mmap_path, mmap_mode="r")

    model = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=1,
        random_state=seed
    )

    labels = model.fit_predict(X)
    inertia = float(model.inertia_)

    return {
        "labels": labels,
        "inertia": inertia,
        "seed": seed
    }


def run_clustering_parallel(X_scaled):
    with tempfile.TemporaryDirectory(prefix="kmeans_eucl_") as mmap_dir:
        mmap_path = os.path.join(mmap_dir, "X_scaled.mmap")
        dump(X_scaled, mmap_path)

        rng = np.random.RandomState(RANDOM_STATE)
        seeds = rng.randint(0, 100_000, size=N_INIT).tolist()

        results = Parallel(n_jobs=N_WORKERS, backend="loky", verbose=0)(
            delayed(_single_init_worker)(
                mmap_path=mmap_path,
                k=K,
                max_iter=MAX_ITER,
                seed=s
            )
            for s in seeds
        )

    best = min(results, key=lambda r: r["inertia"])

    print(f"Euclidean clustering complete.")
    print(f"Best inertia: {best['inertia']:.2f}")
    print(f"Best seed: {best['seed']}")

    return best["labels"]


X_raw, df_profiles = load_profiles(INPUT_CSV)

X_scaled = normalise_profiles(X_raw)

labels = run_clustering_parallel(X_scaled)

df_metrics = pd.read_csv(CLUSTERED_METRICS_CSV)

df_clusters = df_profiles[["house_id", "date"]].copy()

df_clusters["euklidean_cluster_id"] = labels + 1

df_out = df_metrics.merge(
    df_clusters,
    on=["house_id", "date"],
    how="left",
    validate="one_to_one"
)

assert df_out["euklidean_cluster_id"].notna().all(), (
    "Δεν βρέθηκε euclidean cluster για μία ή περισσότερες γραμμές metrics."
)

df_out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved → '{OUTPUT_CSV}'")
# %%
