# %% [markdown]
# # 3. Ομαδοποίηση Ημερήσιων Προφίλ Κατανάλωσης
#
# Αυτό το notebook διαβάζει το
# `2_targeted_period_imputated_daily_profiles.csv`, όπου κάθε γραμμή
# αντιπροσωπεύει μία κατοικία σε μία ημέρα και οι 96 στήλες ώρας περιέχουν το
# συμπληρωμένο 15λεπτο προφίλ συνολικής κατανάλωσης.
#
# Κανονικοποιεί κάθε ημερήσιο προφίλ και ομαδοποιεί τα σχήματα των προφίλ με
# time-series k-means βασισμένο σε DTW. Αρχικά αξιολογεί υποψήφιες τιμές `k`
# από 2 έως 20 με βάση inertia, precomputed DTW silhouette score και
# peak-match score. Στη συνέχεια εκπαιδεύει το τελικό μοντέλο με `k=8`,
# αντιστοιχίζει κάθε κατοικία-ημέρα σε cluster κατανάλωσης και σχεδιάζει raw
# και κανονικοποιημένα προφίλ ανά cluster με DTW barycenters.
#
# Το κύριο αποτέλεσμα είναι το `3_daily_profiles_CLUSTERED.csv`, που κρατά τον
# πίνακα ημερήσιων προφίλ και προσθέτει το `load_cluster_id`. Αποθηκεύει επίσης
# το γράφημα αξιολόγησης του k-sweep και το αντίστοιχο metrics CSV στο
# `config.OUTPUT_PATH`.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config
from helpers import (
    load_daily_profiles,
    normalise_daily_profiles,
    run_dtw_profile_clustering_parallel,
    save_daily_profile_clusters,
)

import multiprocessing
multiprocessing.freeze_support()

import time
import warnings
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from numba import njit, prange
from joblib import Parallel, delayed, dump, load
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from threadpoolctl import threadpool_limits
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.barycenters import dtw_barycenter_averaging

K_MIN              = 2
K_MAX              = 20
SAKOE_CHIBA_RADIUS = 6
DTW_MAX_ITER       = 10
RANDOM_STATE       = 42
N_INIT_SWEEP       = 3
N_CPU              = os.cpu_count() or 1


def build_numba_dtw():
    @njit(parallel=True, fastmath=True, cache=True)
    def dtw_matrix(X: np.ndarray, radius: int) -> np.ndarray:
        n, T = X.shape
        D    = np.zeros((n, n), dtype=np.float32)
        INF  = np.float64(1e18)

        for i in prange(n):
            for j in range(i + 1, n):
                cost       = np.full((T + 1, T + 1), INF)
                cost[0, 0] = 0.0

                for t in range(1, T + 1):
                    lo = max(1, t - radius)
                    hi = min(T, t + radius)

                    for s in range(lo, hi + 1):
                        diff = X[i, t - 1] - X[j, s - 1]
                        c    = diff * diff

                        best = cost[t - 1, s - 1]
                        if cost[t - 1, s] < best:
                            best = cost[t - 1, s]
                        if cost[t, s - 1] < best:
                            best = cost[t, s - 1]

                        cost[t, s] = c + best

                d       = np.sqrt(cost[T, T])
                D[i, j] = d
                D[j, i] = d
        return D

    return dtw_matrix


def warmup_dtw(dtw_fn, radius: int) -> None:
    _x = np.ascontiguousarray(np.random.rand(4, 96))
    with threadpool_limits(limits=1):
        dtw_fn(_x, radius)


def precompute_dtw_matrix(
    X_scaled: np.ndarray,
    radius: int,
    mmap_dir: str,
    dtw_fn,
) -> tuple[np.ndarray, str]:
    X_2d = np.ascontiguousarray(X_scaled[:, :, 0])
    with threadpool_limits(limits=1):
        D = dtw_fn(X_2d, radius)

    mmap_path = os.path.join(mmap_dir, "dtw_matrix.mmap")
    dump(D, mmap_path)
    return D, mmap_path


def build_xtick_labels(n_steps: int = 96, hour_step: int = 4) -> tuple:
    positions, labels = [], []
    for h in range(0, 24, hour_step):
        positions.append(h * 4)
        labels.append(f"{h:02d}:00")
    if (n_steps - 1) not in positions:
        positions.append(n_steps - 1)
        labels.append("23:45")
    return positions, labels

dtw_fn = build_numba_dtw()
warmup_dtw(dtw_fn, radius=SAKOE_CHIBA_RADIUS)

def _pms_single(
    profile_flat: np.ndarray,
    centroid_flat: np.ndarray,
    prominence_threshold: float = 0.3,
    timing_tolerance: int = 4,
) -> float:
    c_peaks, c_props = find_peaks(centroid_flat, prominence=prominence_threshold)

    if len(c_peaks) == 0:
        corr = np.corrcoef(profile_flat, centroid_flat)[0, 1]
        return float(max(0.0, (corr + 1.0) / 2.0))

    p_peaks, _ = find_peaks(profile_flat, prominence=prominence_threshold * 0.5)

    matched_w = 0.0
    total_w   = 0.0

    for ci, cp in zip(c_peaks, c_props["prominences"]):
        total_w += cp
        if len(p_peaks) == 0:
            continue

        dists    = np.abs(p_peaks - ci)
        best_idx = int(np.argmin(dists))

        if dists[best_idx] <= timing_tolerance:
            pi       = p_peaks[best_idx]
            amp_diff = abs(profile_flat[pi] - centroid_flat[ci])
            amp_sim  = max(0.0, 1.0 - amp_diff / (abs(centroid_flat[ci]) + 1e-9))
            t_score  = 1.0 - dists[best_idx] / (timing_tolerance + 1)
            matched_w += cp * (0.6 * amp_sim + 0.4 * t_score)

    return float(matched_w / (total_w + 1e-9))


def _pms_batch(
    profiles_flat: np.ndarray,
    labels: np.ndarray,
    centroids_flat: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return np.array(
        [_pms_single(profiles_flat[i], centroids_flat[labels[i]]) for i in indices],
        dtype=np.float32,
    )


def compute_average_pms_parallel(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_jobs: int,
) -> float:
    profiles_flat  = np.ascontiguousarray(X_scaled[:, :, 0])
    centroids_flat = np.ascontiguousarray(centroids[:, :, 0])
    batches        = np.array_split(np.arange(len(labels)), max(1, n_jobs))

    batch_scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_pms_batch)(profiles_flat, labels, centroids_flat, b)
        for b in batches if len(b) > 0
    )
    return float(np.concatenate(batch_scores).mean())


def _fit_single_k(
    k: int,
    X_scaled: np.ndarray,
    mmap_path: str,
    radius: int,
    max_iter: int,
    n_init: int,
    random_state: int,
    n_jobs_pms: int,
) -> dict:
    t0 = time.perf_counter()

    with threadpool_limits(limits=1):
        model = TimeSeriesKMeans(
            n_clusters    = k,
            metric        = "dtw",
            metric_params = {"sakoe_chiba_radius": radius},
            max_iter      = max_iter,
            n_init        = n_init,
            random_state  = random_state,
            n_jobs        = 1,
            verbose       = False,
        )
        labels    = model.fit_predict(X_scaled)
        centroids = model.cluster_centers_
        inertia   = float(model.inertia_)

    D   = load(mmap_path, mmap_mode="r")
    sil = float(silhouette_score(D, labels, metric="precomputed"))
    avg_pms = compute_average_pms_parallel(X_scaled, labels, centroids, n_jobs=n_jobs_pms)

    return {
        "k":          k,
        "inertia":    inertia,
        "silhouette": sil,
        "avg_pms":    avg_pms,
        "elapsed_s":  time.perf_counter() - t0,
    }


def evaluate_all_k_parallel(
    X_scaled: np.ndarray,
    mmap_path: str,
    k_range: range,
    radius: int,
) -> pd.DataFrame:
    k_values  = list(k_range)
    n_k       = len(k_values)
    n_workers = min(N_CPU, n_k)
    n_jobs_pms = max(1, min(2, N_CPU // n_workers))

    raw_results = Parallel(
        n_jobs=n_workers,
        backend="loky",
        return_as="generator_unordered",
    )(
        delayed(_fit_single_k)(
            k            = k,
            X_scaled     = X_scaled,
            mmap_path    = mmap_path,
            radius       = radius,
            max_iter     = DTW_MAX_ITER,
            n_init       = N_INIT_SWEEP,
            random_state = RANDOM_STATE,
            n_jobs_pms   = n_jobs_pms,
        )
        for k in k_values
    )

    collected = []
    with tqdm(total=n_k, desc="Εκπαίδευση μοντέλων", unit="k", ncols=90, colour="cyan") as pbar:
        for res in raw_results:
            collected.append(res)
            pbar.set_postfix(
                k=res["k"],
                sil=f"{res['silhouette']:.4f}",
                pms=f"{res['avg_pms']:.4f}",
                t=f"{res['elapsed_s']:.1f}s",
            )
            pbar.update(1)

    return pd.DataFrame(collected).sort_values("k").reset_index(drop=True)


def plot_sweep_results(results: pd.DataFrame, chosen_k: int) -> None:
    df = results.copy()
    df["inertia_drop_%"] = (
        (df["inertia"] - df["inertia"].shift(-1)) / df["inertia"] * 100
    ).fillna(0.0)

    ks = df["k"].values

    BG        = "#ffffff"
    PANEL_BG  = "#f9f9f9"
    GRID_COL  = "#e0e4ef"
    LINE_COL  = "#4fc3f7"
    DOT_COL   = "#aaaaaa"
    VLINE_COL = "#ff6b6b"
    TITLE_COL = "#1a1a2e"
    LABEL_COL = "#4a4a5a"

    panels = [
        ("inertia",       "DTW Inertia  ↓",           "Inertia"),
        ("inertia_drop_%","Πτώση Inertia %  ↓",        "Drop %"),
        ("silhouette",    "Silhouette Score  ↑",        "Score"),
        ("avg_pms",       "Avg Peak-Match Score  ↑",   "PMS"),
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 8),
        facecolor=BG,
        gridspec_kw={"hspace": 0.52, "wspace": 0.32},
    )
    axes = axes.flat

    fig.suptitle(
        "Αξιολόγηση Clustering  |  k-sweep",
        fontsize=15, fontweight="bold",
        color=TITLE_COL, y=0.97,
    )

    for ax, (col, title, ylabel) in zip(axes, panels):
        y = df[col].values

        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=LABEL_COL, labelsize=9)
        ax.xaxis.label.set_color(LABEL_COL)
        ax.yaxis.label.set_color(LABEL_COL)

        ax.grid(axis="y", color=GRID_COL, linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)

        ax.plot(ks, y, color=LINE_COL, linewidth=2.2, zorder=3)
        ax.scatter(ks, y, color=DOT_COL, s=28, zorder=4, linewidths=0)

        chosen_y = float(df.loc[df["k"] == chosen_k, col].iloc[0])
        ax.axvline(chosen_k, color=VLINE_COL, linewidth=1.4,
                   linestyle="--", zorder=2, alpha=0.85)
        ax.axhline(chosen_y, color=VLINE_COL, linewidth=1.4,
                   linestyle="--", zorder=2, alpha=0.85)
        ax.scatter([chosen_k], [chosen_y],
                   color=VLINE_COL, s=70, zorder=5, linewidths=0)

        ax.set_title(title, fontsize=10.5, fontweight="bold",
                     color=TITLE_COL, pad=8)
        ax.set_xlabel("k  (αριθμός συστάδων)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = config.OUTPUT_PATH + "/3_clustering_sweep_metrics.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Το γράφημα αποθηκεύτηκε επιτυχώς στο '{out_png}'")

    out_csv = config.OUTPUT_PATH + "/3_clustering_evaluation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Το CSV αποθηκεύτηκε επιτυχώς στο '{out_csv}'")

print(f"\nΑξιολόγηση k={K_MIN}–{K_MAX} | Z-score | CPUs={N_CPU}")

X_raw, _meta, df_raw, interval_cols = load_daily_profiles(config.DATA_PATH + "/2_targeted_period_imputated_daily_profiles.csv")
X_scaled = normalise_daily_profiles(X_raw)

with tempfile.TemporaryDirectory(prefix="dtw_sweep_") as mmap_dir:
    print("\nΥπολογισμός πίνακα αποστάσεων DTW…")
    _D, mmap_path = precompute_dtw_matrix(X_scaled, SAKOE_CHIBA_RADIUS, mmap_dir, dtw_fn)

    sweep_results = evaluate_all_k_parallel(
        X_scaled  = X_scaled,
        mmap_path = mmap_path,
        k_range   = range(K_MIN, K_MAX + 1),
        radius    = SAKOE_CHIBA_RADIUS,
    )

plot_sweep_results(sweep_results, chosen_k=8)

# %%
K              = 8
MAX_ITER_FINAL = 15
N_INIT_FINAL   = 5
N_WORKERS      = min(N_CPU, N_INIT_FINAL)

XTICK_HOUR_STEP = 4

INPUT_CSV  = config.DATA_PATH + "/2_targeted_period_imputated_daily_profiles.csv"
OUTPUT_CSV = config.DATA_PATH + "/3_daily_profiles_CLUSTERED.csv"

FIG_SIZE       = (16, 6)
SUPTITLE_FS    = 14
SUBTITLE_FS    = 12
LABEL_FS       = 11
TICK_FS        = 10

PROFILE_COLOR  = "#9E9E9E"
PROFILE_ALPHA  = 0.08
PROFILE_LW     = 0.5
MEAN_COLOR     = "#1565C0"
MEAN_LW        = 3.2
CENTROID_COLOR = "#E65100"
CENTROID_LW    = 3.2

DBA_MAX_ITER   = 10
SAKOE_RADIUS   = 6


def _profiles_to_segments(profiles: np.ndarray) -> np.ndarray:
    N, T   = profiles.shape
    x_axis = np.arange(T, dtype=np.float64)
    xs     = np.broadcast_to(x_axis, (N, T))
    return np.stack([xs, profiles], axis=-1)


def _style_ax(ax, title, ylabel, tick_pos, tick_lbl, y_min, y_max):
    pad = (y_max - y_min) * 0.05
    ax.set_xlim(0, 95)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(title, fontsize=SUBTITLE_FS, fontweight="bold", pad=8)
    ax.set_xlabel("Ώρα της Ημέρας", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=TICK_FS)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.grid(axis="y", linestyle=":",  linewidth=0.4, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cluster(load_cluster_id, raw_members, tick_pos, tick_lbl):
    N        = raw_members.shape[0]
    raw_mean = raw_members.mean(axis=0)

    scaler       = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
    norm_members = scaler.fit_transform(raw_members[:, :, np.newaxis])[:, :, 0]

    dba_centroid = dtw_barycenter_averaging(
        norm_members[:, :, np.newaxis],
        max_iter      = DBA_MAX_ITER,
        metric_params = {"sakoe_chiba_radius": SAKOE_RADIUS},
    )[:, 0]

    segs_raw  = _profiles_to_segments(raw_members)
    segs_norm = _profiles_to_segments(norm_members)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.suptitle(
        f"Cluster {load_cluster_id} — N = {N} ημερήσια προφίλ",
        fontsize=SUPTITLE_FS + 1, fontweight="bold", y=1.01,
    )

    x_axis = np.arange(96)

    ax_left.add_collection(LineCollection(
        segs_raw, colors=PROFILE_COLOR, linewidth=PROFILE_LW, alpha=PROFILE_ALPHA, zorder=1,
    ))
    ax_left.plot(x_axis, raw_mean, color=MEAN_COLOR, linewidth=MEAN_LW, zorder=2)
    _style_ax(ax_left, "Κατανάλωση & Αριθμητικός Μέσος",
              "Κατανάλωση Ενέργειας",
              tick_pos, tick_lbl, raw_members.min(), raw_members.max())
    ax_left.legend(handles=[
        Line2D([0], [0], color=PROFILE_COLOR, lw=1.5, alpha=0.5, label=f"Ατομικά προφίλ (N={N})"),
        Line2D([0], [0], color=MEAN_COLOR, lw=MEAN_LW, label="Αριθμητικός μέσος"),
    ], fontsize=TICK_FS, loc="upper right")

    ax_right.add_collection(LineCollection(
        segs_norm, colors=PROFILE_COLOR, linewidth=PROFILE_LW, alpha=PROFILE_ALPHA, zorder=1,
    ))
    ax_right.plot(x_axis, dba_centroid, color=CENTROID_COLOR, linewidth=CENTROID_LW, zorder=2)
    _style_ax(ax_right, "Κανονικοποιημένη Μορφή & Κέντρο Βάρους DTW (DBA)",
              "Z-score (μονάδες σ)",
              tick_pos, tick_lbl, -1, 10.0)
    ax_right.set_ylim(-1, 10)
    ax_right.axhline(0, color="#BDBDBD", linewidth=0.8, linestyle="--", zorder=0)
    ax_right.legend(handles=[
        Line2D([0], [0], color=PROFILE_COLOR, lw=1.5, alpha=0.5, label=f"Κανονικοποιημένα προφίλ (N={N})"),
        Line2D([0], [0], color=CENTROID_COLOR, lw=CENTROID_LW, label="Κέντρο βάρους DTW (DBA)"),
    ], fontsize=TICK_FS, loc="upper right")

    fig.tight_layout()
    plt.show()

print(f"Clustering Ημερήσιων Προφίλ Κατανάλωσης | k={K}")

X_raw, _meta, df_raw, interval_cols = load_daily_profiles(INPUT_CSV)
X_scaled = normalise_daily_profiles(X_raw)

with tempfile.TemporaryDirectory(prefix="cluster_k8_") as mmap_dir:
    labels_0, centroids, inertia = run_dtw_profile_clustering_parallel(
        X_scaled     = X_scaled,
        mmap_dir     = mmap_dir,
        k            = K,
        radius       = SAKOE_CHIBA_RADIUS,
        max_iter     = MAX_ITER_FINAL,
        n_init       = N_INIT_FINAL,
        random_state = RANDOM_STATE,
        n_workers    = N_WORKERS,
    )

df_out = save_daily_profile_clusters(df_raw, labels_0, OUTPUT_CSV)

tick_pos, tick_lbl = build_xtick_labels(hour_step=XTICK_HOUR_STEP)
df_raw_vals        = pd.DataFrame(X_raw, index=df_out.index, columns=interval_cols)

for load_cluster_id in range(1, K + 1):
    mask        = df_out["load_cluster_id"] == load_cluster_id
    raw_members = df_raw_vals.loc[mask].values

    if raw_members.shape[0] == 0:
        print(f"Προειδοποίηση: Η συστάδα {load_cluster_id} είναι άδεια — παραλείπεται.")
        continue

    plot_cluster(
        load_cluster_id  = load_cluster_id,
        raw_members = raw_members,
        tick_pos    = tick_pos,
        tick_lbl    = tick_lbl,
    )
# %%
