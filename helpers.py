import os
from collections.abc import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


DEFAULT_META_COLS = ("house_id", "date")
DEFAULT_FLEXIBILITY_FEATURE_COLS = ("SEF", "PSS", "PAR", "Slack_i")


def load_daily_profiles(
    path: str,
    meta_cols: Sequence[str] = DEFAULT_META_COLS,
    expected_intervals: int = 96,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[str]]:
    df_raw = pd.read_csv(path)

    missing_meta = [c for c in meta_cols if c not in df_raw.columns]
    assert not missing_meta, (
        f"Λείπουν οι στήλες metadata: {missing_meta}"
    )

    interval_cols = [c for c in df_raw.columns if c not in meta_cols]
    assert len(interval_cols) == expected_intervals, (
        f"Αναμένονταν {expected_intervals} χρονικά διαστήματα, "
        f"βρέθηκαν {len(interval_cols)}."
    )

    meta = df_raw[list(meta_cols)].copy()
    X_raw = df_raw[interval_cols].values.astype(np.float64)

    print(f"Φόρτωση: {X_raw.shape[0]} προφίλ × {X_raw.shape[1]} διαστήματα.")
    return X_raw, meta, df_raw, interval_cols


def normalise_daily_profiles(X_raw: np.ndarray) -> np.ndarray:
    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
    return scaler.fit_transform(X_raw[:, :, np.newaxis])


def _single_dtw_profile_init_worker(
    mmap_path: str,
    k: int,
    radius: int,
    max_iter: int,
    seed: int,
) -> dict:
    X = load(mmap_path, mmap_mode="r")
    with threadpool_limits(limits=1):
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            metric_params={"sakoe_chiba_radius": radius},
            max_iter=max_iter,
            n_init=1,
            random_state=seed,
            n_jobs=1,
            verbose=False,
        )
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        centers = model.cluster_centers_.copy()

    return {"labels": labels, "inertia": inertia, "centers": centers, "seed": seed}


def run_dtw_profile_clustering_parallel(
    X_scaled: np.ndarray,
    mmap_dir: str,
    k: int,
    radius: int = 6,
    max_iter: int = 15,
    n_init: int = 5,
    random_state: int = 42,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    mmap_path = os.path.join(mmap_dir, f"X_scaled_k{k}.mmap")
    dump(X_scaled, mmap_path)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_init)

    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 100_000, size=n_init).tolist()

    results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
        delayed(_single_dtw_profile_init_worker)(
            mmap_path=mmap_path,
            k=k,
            radius=radius,
            max_iter=max_iter,
            seed=s,
        )
        for s in seeds
    )

    best = min(results, key=lambda r: r["inertia"])
    print("\nTo clustering ολοκληρώθηκε.\n")
    return best["labels"], best["centers"], best["inertia"]


def save_daily_profile_clusters(
    df_raw: pd.DataFrame,
    labels_0: np.ndarray,
    output_path: str,
    cluster_col: str = "load_cluster_id",
) -> pd.DataFrame:
    df_out = df_raw.copy()
    df_out[cluster_col] = labels_0 + 1
    df_out.to_csv(output_path, index=False)
    print(f"Το dataframe των clustered daily profiles αποθηκεύτηκε επιτυχώς στο '{output_path}'")
    return df_out


def load_flexibility_metrics(
    path: str,
    feature_cols: Sequence[str] = DEFAULT_FLEXIBILITY_FEATURE_COLS,
    meta_cols: Sequence[str] = DEFAULT_META_COLS,
) -> tuple[np.ndarray, pd.DataFrame]:
    df_raw = pd.read_csv(path)

    missing_meta = [c for c in meta_cols if c not in df_raw.columns]
    assert not missing_meta, (
        f"Λείπουν οι στήλες metadata: {missing_meta}"
    )

    missing_features = [c for c in feature_cols if c not in df_raw.columns]
    assert not missing_features, (
        f"Λείπουν μία ή περισσότερες στήλες χαρακτηριστικών: {missing_features}"
    )

    print(f"Φόρτωση: {len(df_raw)} εγγραφές × {len(feature_cols)} χαρακτηριστικά.")
    return df_raw[list(feature_cols)].values.astype(np.float64), df_raw


def scale_flexibility_features(X_raw: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X_raw)


def run_flexibility_kmeans_clustering(
    X_scaled: np.ndarray,
    k: int,
    max_iter: int = 300,
    n_init: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    model = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(X_scaled)
    return labels, model


def save_flexibility_clusters(
    df_raw: pd.DataFrame,
    labels_0: np.ndarray,
    output_path: str,
    cluster_col: str = "flexibility_cluster_id",
) -> pd.DataFrame:
    df_out = df_raw.copy()
    df_out[cluster_col] = labels_0 + 1
    df_out.to_csv(output_path, index=False)
    return df_out


def default_cluster_method_labels(
    load_cluster_col: str,
    flexibility_cluster_col: str,
) -> dict[str, str]:
    return {
        load_cluster_col: "Load-based",
        flexibility_cluster_col: "Flexibility-aware",
    }


def default_cluster_method_colors(
    load_cluster_col: str,
    flexibility_cluster_col: str,
) -> dict[str, str]:
    return {
        load_cluster_col: "#4C72B0",
        flexibility_cluster_col: "#DD8452",
    }


def build_dr_targeting_scenarios(
    df: pd.DataFrame,
    methods: dict[str, str],
    target_col: str = "delta_load",
) -> tuple[pd.DataFrame, dict[str, list[tuple[float, float]]]]:
    total_days = len(df)
    total_target = df[target_col].sum()

    scenario_records = []

    def scenario_stats(mask, scenario: str, method_label: str) -> dict:
        sel = df[mask]
        captured_target = sel[target_col].sum()
        return dict(
            method=method_label,
            scenario=scenario,
            selected_days=len(sel),
            total_days=total_days,
            pct_days_selected=len(sel) / total_days,
            captured_delta_load=captured_target,
            total_delta_load=total_target,
            pct_dr_captured=captured_target / total_target,
        )

    gain_records = {}
    for col, label in methods.items():
        cluster_means = df.groupby(col)[target_col].mean().sort_values(ascending=False)
        ranked_clusters = cluster_means.index.tolist()

        scenario_records.append(
            scenario_stats(df[col] == ranked_clusters[0], "A: Top 1 cluster", label)
        )

        top2 = ranked_clusters[:2]
        scenario_records.append(
            scenario_stats(df[col].isin(top2), "B: Top 2 clusters", label)
        )

        selected_c = []
        for cid in ranked_clusters:
            selected_c.append(cid)
            if df[col].isin(selected_c).sum() / total_days >= 0.30:
                break
        scenario_records.append(
            scenario_stats(df[col].isin(selected_c), "C: Top 30% days", label)
        )

        selected_d = []
        for cid in ranked_clusters:
            selected_d.append(cid)
            if df[col].isin(selected_d).sum() / total_days >= 0.50:
                break
        scenario_records.append(
            scenario_stats(df[col].isin(selected_d), "D: Top 50% days", label)
        )

        rows = []
        cum_days = 0
        cum_target = 0.0
        for cid in ranked_clusters:
            mask = df[col] == cid
            cum_days += mask.sum()
            cum_target += df.loc[mask, target_col].sum()
            rows.append((cum_days / total_days, cum_target / total_target))
        gain_records[label] = rows

    return pd.DataFrame(scenario_records), gain_records


def plot_dr_targeting_scenarios(
    df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    gain_records: dict[str, list[tuple[float, float]]],
    methods: dict[str, str],
    colors: dict[str, str],
    output_png: str,
    target_col: str = "delta_load",
    title_suffix: str = "",
    show: bool = True,
) -> None:
    title = "DR Targeting Experiment - Scenario Analysis"
    if title_suffix:
        title = f"{title} | {title_suffix}"

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    for plot_i, (metric, ylabel, subtitle) in enumerate([
        ("pct_days_selected", "Fraction of Days Selected", "Days Selected per Scenario"),
        ("pct_dr_captured", "Fraction of DR Captured", "DR Potential Captured per Scenario"),
    ]):
        ax = fig.add_subplot(gs[0, plot_i])
        scenarios = scenarios_df["scenario"].unique()
        x = np.arange(len(scenarios))
        width = 0.35
        for i, (col, label) in enumerate(methods.items()):
            sub = scenarios_df[scenarios_df["method"] == label].set_index("scenario")
            ax.bar(
                x + i * width,
                [sub.loc[s, metric] for s in scenarios],
                width=width,
                color=colors[col],
                label=label,
                alpha=0.85,
            )
        ax.set_title(subtitle, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([s.split(":")[0] for s in scenarios])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    scenarios = scenarios_df["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.35
    for i, (col, label) in enumerate(methods.items()):
        sub = scenarios_df[scenarios_df["method"] == label].set_index("scenario")
        ratios = [
            sub.loc[s, "pct_dr_captured"] / sub.loc[s, "pct_days_selected"]
            for s in scenarios
        ]
        ax.bar(x + i * width, ratios, width=width, color=colors[col], label=label, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (ratio=1)")
    ax.set_title("DR Efficiency\n(DR captured / days used)", fontsize=11)
    ax.set_ylabel("Efficiency Ratio")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([s.split(":")[0] for s in scenarios])
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for _col, label in methods.items():
        pts = gain_records[label]
        xs = [0] + [p[0] for p in pts]
        ys = [0] + [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=colors[_col], label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_title("Cumulative Gain Curve\n(by cluster rank)", fontsize=11)
    ax.set_xlabel("Fraction of Days Selected")
    ax.set_ylabel("Fraction of DR Captured")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[1, 1])
    for col, label in methods.items():
        means = df.groupby(col)[target_col].mean().sort_values(ascending=False).values
        ax.plot(range(1, len(means) + 1), means, marker="o", color=colors[col], label=label, linewidth=2)
    ax.set_title("Cluster Ranking\n(mean delta_load, descending)", fontsize=11)
    ax.set_xlabel("Cluster Rank")
    ax.set_ylabel("Mean delta_load")
    ax.set_xticks(range(1, max(df[col].nunique() for col in methods) + 1))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    tbl_data = []
    col_labels = ["Method", "Scenario", "% Days", "% DR"]
    for _, row in scenarios_df.iterrows():
        tbl_data.append([
            row["method"].replace("Load-based", "Load").replace("Flexibility-aware", "Flex"),
            row["scenario"].split(":")[0],
            f"{row['pct_days_selected']:.1%}",
            f"{row['pct_dr_captured']:.1%}",
        ])
    tbl = ax.table(cellText=tbl_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.4)
    ax.set_title("Summary Table", fontsize=11, pad=12)

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_dr_targeting_analysis(
    df: pd.DataFrame,
    load_cluster_col: str,
    flexibility_cluster_col: str,
    output_csv: str,
    output_png: str,
    target_col: str = "delta_load",
    title_suffix: str = "",
    show: bool = True,
) -> pd.DataFrame:
    methods = default_cluster_method_labels(load_cluster_col, flexibility_cluster_col)
    colors = default_cluster_method_colors(load_cluster_col, flexibility_cluster_col)

    scenarios_df, gain_records = build_dr_targeting_scenarios(df, methods, target_col=target_col)
    scenarios_df.to_csv(output_csv, index=False)

    plot_dr_targeting_scenarios(
        df=df,
        scenarios_df=scenarios_df,
        gain_records=gain_records,
        methods=methods,
        colors=colors,
        output_png=output_png,
        target_col=target_col,
        title_suffix=title_suffix,
        show=show,
    )

    return scenarios_df


def run_kruskal_dunn_analysis(
    df: pd.DataFrame,
    load_cluster_col: str,
    flexibility_cluster_col: str,
    output_kw_csv: str,
    output_sig_pairs_csv: str,
    output_kw_png: str,
    dunn_heatmap_prefix: str,
    metrics: Sequence[str] = ("delta_load", "SEF", "PSS", "PAR"),
    alpha: float = 0.05,
    title_suffix: str = "",
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import scikit_posthocs as sp

    methods = default_cluster_method_labels(load_cluster_col, flexibility_cluster_col)

    kw_records = []
    for col, label in methods.items():
        for metric in metrics:
            groups = [grp[metric].values for _, grp in df.groupby(col)]
            stat, p = stats.kruskal(*groups)
            kw_records.append(dict(
                method=label,
                metric=metric,
                kruskal_stat=stat,
                p_value=p,
                significant=(p < alpha),
            ))

    kw_df = pd.DataFrame(kw_records)
    kw_df.to_csv(output_kw_csv, index=False)

    dunn_results = {}
    sig_pairs_records = []

    for col, label in methods.items():
        for metric in metrics:
            row = kw_df[(kw_df["method"] == label) & (kw_df["metric"] == metric)].iloc[0]
            if not row["significant"]:
                continue
            dunn = sp.posthoc_dunn(df, val_col=metric, group_col=col, p_adjust="holm")
            dunn_results[(col, metric)] = dunn
            clusters = dunn.columns.tolist()
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    p_val = dunn.loc[clusters[i], clusters[j]]
                    if p_val < alpha:
                        sig_pairs_records.append(dict(
                            method=label,
                            metric=metric,
                            cluster_a=clusters[i],
                            cluster_b=clusters[j],
                            dunn_p_value=p_val,
                        ))

    sig_pairs_df = pd.DataFrame(
        sig_pairs_records,
        columns=["method", "metric", "cluster_a", "cluster_b", "dunn_p_value"],
    )
    sig_pairs_df.to_csv(output_sig_pairs_csv, index=False)

    title = "Kruskal-Wallis p-values per Metric and Method"
    if title_suffix:
        title = f"{title} | {title_suffix}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, (col, label) in zip(axes, methods.items()):
        sub = kw_df[kw_df["method"] == label].set_index("metric")
        p_vals = sub.loc[list(metrics), "p_value"].values
        bars = ax.bar(
            metrics,
            p_vals,
            color=["#2ca02c" if p < alpha else "#d62728" for p in p_vals],
            alpha=0.85,
        )
        ax.axhline(alpha, color="black", linestyle="--", linewidth=1, label=f"alpha = {alpha}")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("p-value (log scale)")
        ax.legend(fontsize=8)
        for bar, p in zip(bars, p_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{p:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(output_kw_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    for col, label in methods.items():
        available = [m for m in metrics if (col, m) in dunn_results]
        if not available:
            print(f"No significant KW results for {label} — skipping Dunn heatmaps.")
            continue

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
        if len(available) == 1:
            axes = [axes]
        heatmap_title = f"Dunn Test p-value Matrices (Holm correction) — {label}"
        if title_suffix:
            heatmap_title = f"{heatmap_title} | {title_suffix}"
        fig.suptitle(heatmap_title, fontsize=13, fontweight="bold")

        for ax, metric in zip(axes, available):
            dunn = dunn_results[(col, metric)]
            mat = dunn.values.astype(float)
            n = len(dunn.columns)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(n))
            ax.set_xticklabels(dunn.columns, fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(dunn.index, fontsize=8)
            ax.set_title(metric, fontsize=11)
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    txt = f"{val:.3f}" if not np.isnan(val) else ""
                    color = "white" if val < 0.3 or val > 0.7 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("p-value", fontsize=8)

        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace("-", "_")
        plt.savefig(f"{dunn_heatmap_prefix}_{safe_label}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return kw_df, sig_pairs_df
