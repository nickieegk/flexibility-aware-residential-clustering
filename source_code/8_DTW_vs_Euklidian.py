# %% [markdown]
# # 8. Σύγκριση DTW και Euclidean Load-based Clustering
#
# Αυτό το notebook διαβάζει το `7_days_metrics_CLUSTERED_three_ways.csv`, το
# οποίο περιέχει μία γραμμή ανά κατοικία-ημέρα με `load_cluster_id` από το
# DTW load-based clustering και `euklidean_cluster_id` από το Euclidean
# load-based clustering, καθώς και demand-response metrics όπως `delta_load`,
# `SEF`, `PSS` και `PAR`.
#
# Συγκρίνει τις δύο load-based προσεγγίσεις clustering από διάφορες πλευρές:
# summary statistics και box plots του `delta_load`, inter-cluster variance στο
# δυναμικό demand response, targeting scenarios που ταξινομούν τα clusters με
# βάση το captured `delta_load`, και μη παραμετρικούς ελέγχους σημαντικότητας
# μεταξύ clusters.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

INPUT_CSV = config.DATA_PATH + "/7_days_metrics_CLUSTERED_three_ways.csv"

df = pd.read_csv(INPUT_CSV)

METHODS = {
    "load_cluster_id": "DTW load-based",
    "euklidean_cluster_id": "Euclidean load-based",
}

COLORS = {
    "load_cluster_id": "#4C72B0",
    "euklidean_cluster_id": "#55A868",
}

REQUIRED_COLS = [
    "load_cluster_id",
    "euklidean_cluster_id",
    "delta_load",
    "SEF",
    "PSS",
    "PAR",
]

assert all(c in df.columns for c in REQUIRED_COLS), (
    f"Λείπουν μία ή περισσότερες απαιτούμενες στήλες: {REQUIRED_COLS}"
)


def cluster_stats(df, cluster_col):
    return (
        df.groupby(cluster_col)["delta_load"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            variance="var",
            min="min",
            max="max",
        )
        .reset_index()
        .rename(columns={cluster_col: "cluster_id"})
        .assign(method=METHODS[cluster_col])
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )


dtw_stats = cluster_stats(df, "load_cluster_id")
euclidean_stats = cluster_stats(df, "euklidean_cluster_id")

combined = pd.concat([dtw_stats, euclidean_stats], ignore_index=True)
combined = combined[["method", "cluster_id", "count", "mean", "median", "std", "variance", "min", "max"]]

combined.to_csv(
    config.OUTPUT_PATH + "/8_DTW_vs_Euklidian_delta_load_clusters_comparison.csv",
    index=False
)

methods = {
    "load_cluster_id": dtw_stats,
    "euklidean_cluster_id": euclidean_stats
}

labels = METHODS
colors = COLORS

metrics = ["mean", "median", "std"]

all_cluster_ids = sorted(
    set(dtw_stats["cluster_id"]).union(set(euclidean_stats["cluster_id"]))
)

x = np.arange(len(all_cluster_ids))
bar_width = 0.35

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "delta_load statistics per cluster — DTW vs Euclidean load-based",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.45,
    wspace=0.35
)

for col, metric in enumerate(metrics):
    ax = fig.add_subplot(gs[0, col])

    for i, (key, stats) in enumerate(methods.items()):
        stats_aligned = (
            stats
            .set_index("cluster_id")
            .reindex(all_cluster_ids)
        )

        offset = (i - 0.5) * bar_width

        ax.bar(
            x + offset,
            stats_aligned[metric],
            width=bar_width,
            color=colors[key],
            label=labels[key],
            alpha=0.85
        )

    ax.set_title(metric.capitalize(), fontsize=11)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("delta_load")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cluster_ids)

    if col == 0:
        ax.legend(fontsize=8)

for col, (key, stats) in enumerate(methods.items()):
    ax = fig.add_subplot(gs[1, col])

    cluster_ids = sorted(df[key].unique())

    data = [
        df.loc[df[key] == cid, "delta_load"].values
        for cid in cluster_ids
    ]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5)
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(colors[key])
        patch.set_alpha(0.75)

    ax.set_title(f"{labels[key]} — box plots", fontsize=11)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("delta_load")

    ax.set_xticks(np.arange(1, len(cluster_ids) + 1))
    ax.set_xticklabels(cluster_ids)

ax = fig.add_subplot(gs[1, 2])

for i, (key, stats) in enumerate(methods.items()):
    stats_aligned = (
        stats
        .set_index("cluster_id")
        .reindex(all_cluster_ids)
    )

    offset = (i - 0.5) * bar_width

    ax.bar(
        x + offset,
        stats_aligned["count"],
        width=bar_width,
        color=colors[key],
        label=labels[key],
        alpha=0.85
    )

ax.set_title("Cluster sizes (# days)", fontsize=11)
ax.set_xlabel("Cluster ID")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(all_cluster_ids)
ax.legend(fontsize=8)

plt.savefig(
    config.OUTPUT_PATH + "/8_DTW_vs_Euklidian_delta_load_clusters_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(combined.to_string(index=False))
# %% [markdown]
# Ανάλυση Διακύμανσης Μεταξύ Clusters
# %%
from scipy import stats

methods = {
    "load_cluster_id":        "DTW load-based",
    "euklidean_cluster_id":   "Euclidean load-based",
}
colors = {
    "load_cluster_id":        "#4C72B0",
    "euklidean_cluster_id":   "#55A868",
}

records = []
for col, label in methods.items():
    grp = df.groupby(col)["delta_load"]
    cluster_means = grp.mean()
    n_per_cluster = grp.count()
    se_per_cluster = grp.std() / np.sqrt(n_per_cluster)
    inter_var = cluster_means.var()
    for cid in cluster_means.index:
        records.append(dict(
            method=label,
            cluster_id=cid,
            cluster_mean=cluster_means[cid],
            cluster_se=se_per_cluster[cid],
            ci95_lower=cluster_means[cid] - 1.96 * se_per_cluster[cid],
            ci95_upper=cluster_means[cid] + 1.96 * se_per_cluster[cid],
            inter_cluster_variance=inter_var,
        ))

results = pd.DataFrame(records)
results.to_csv(config.DATA_PATH + "/8_DTW_vs_Euklidian_inter_cluster_variance.csv", index=False)

summary = (
    results.groupby("method")
    .agg(
        n_clusters=("cluster_id", "count"),
        inter_cluster_variance=("inter_cluster_variance", "first"),
        min_cluster_mean=("cluster_mean", "min"),
        max_cluster_mean=("cluster_mean", "max"),
        range_cluster_means=("cluster_mean", lambda x: x.max() - x.min()),
    )
    .reset_index()
)
print(summary.to_string(index=False))

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    "Inter-cluster Variance Analysis — DTW vs Euclidean load-based",
    fontsize=14,
    fontweight="bold",
    y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

method_keys = list(methods.keys())

for i, col in enumerate(method_keys):
    ax = fig.add_subplot(gs[0, i])
    sub = results[results["method"] == methods[col]].sort_values("cluster_id")
    ax.bar(sub["cluster_id"].astype(str), sub["cluster_mean"], color=colors[col], alpha=0.85, width=0.5)
    ax.set_title(f"{methods[col]}\nCluster Means of delta_load", fontsize=11)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean delta_load")

for i, col in enumerate(method_keys):
    ax = fig.add_subplot(gs[1, i])
    sub = results[results["method"] == methods[col]].sort_values("cluster_id")
    x = np.arange(len(sub))
    ax.errorbar(
        x, sub["cluster_mean"],
        yerr=1.96 * sub["cluster_se"],
        fmt="o", color=colors[col], capsize=6, capthick=1.5,
        markersize=7, linewidth=1.5,
    )
    inter_var = sub["inter_cluster_variance"].iloc[0]
    ax.set_title(f"{methods[col]}\n95% CI — Inter-cluster Var = {inter_var:.4f}", fontsize=11)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean delta_load")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["cluster_id"].astype(str))

plt.savefig(config.DATA_PATH + "/8_DTW_vs_Euklidian_inter_cluster_variance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Πείραμα Στόχευσης Demand Response
# %%
methods = {
    "load_cluster_id":        "DTW load-based",
    "euklidean_cluster_id":   "Euclidean load-based",
}
colors = {
    "load_cluster_id":        "#4C72B0",
    "euklidean_cluster_id":   "#55A868",
}

total_days       = len(df)
total_delta_load = df["delta_load"].sum()

scenario_records = []

for col, label in methods.items():
    cluster_means = df.groupby(col)["delta_load"].mean().sort_values(ascending=False)
    ranked_clusters = cluster_means.index.tolist()

    def scenario_stats(mask, scenario, method_label):
        sel = df[mask]
        return dict(
            method=method_label,
            scenario=scenario,
            selected_days=len(sel),
            total_days=total_days,
            pct_days_selected=len(sel) / total_days,
            captured_delta_load=sel["delta_load"].sum(),
            total_delta_load=total_delta_load,
            pct_dr_captured=sel["delta_load"].sum() / total_delta_load,
        )

    scenario_records.append(scenario_stats(df[col] == ranked_clusters[0], "A: Top 1 cluster", label))

    top2 = ranked_clusters[:2]
    scenario_records.append(scenario_stats(df[col].isin(top2), "B: Top 2 clusters", label))

    # C: προσθέτουμε clusters (ranked) μέχρι οι ημέρες τους να φτάσουν ~30% του dataset
    selected_c = []
    for cid in ranked_clusters:
        selected_c.append(cid)
        if df[col].isin(selected_c).sum() / total_days >= 0.30:
            break
    scenario_records.append(scenario_stats(df[col].isin(selected_c), "C: Top 30% days", label))

    # D: προσθέτουμε clusters (ranked) μέχρι οι ημέρες τους να φτάσουν ~50% του dataset
    selected_d = []
    for cid in ranked_clusters:
        selected_d.append(cid)
        if df[col].isin(selected_d).sum() / total_days >= 0.50:
            break
    scenario_records.append(scenario_stats(df[col].isin(selected_d), "D: Top 50% days", label))

scenarios_df = pd.DataFrame(scenario_records)
scenarios_df.to_csv(config.DATA_PATH + "/8_DTW_vs_Euklidian_dr_targeting_scenarios.csv", index=False)

gain_records = {}
for col, label in methods.items():
    cluster_means  = df.groupby(col)["delta_load"].mean().sort_values(ascending=False)
    ranked_clusters = cluster_means.index.tolist()
    rows = []
    cum_days = 0
    cum_dr   = 0.0
    for cid in ranked_clusters:
        mask = df[col] == cid
        cum_days += mask.sum()
        cum_dr   += df.loc[mask, "delta_load"].sum()
        rows.append((cum_days / total_days, cum_dr / total_delta_load))
    gain_records[label] = rows

fig = plt.figure(figsize=(18, 12))
fig.suptitle("DR Targeting Experiment — DTW vs Euclidean load-based", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

for plot_i, (metric, ylabel, title) in enumerate([
    ("pct_days_selected", "Fraction of Days Selected", "Days Selected per Scenario"),
    ("pct_dr_captured",   "Fraction of DR Captured",  "DR Potential Captured per Scenario"),
]):
    ax = fig.add_subplot(gs[0, plot_i])
    scenarios = scenarios_df["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.35
    for i, (col, label) in enumerate(methods.items()):
        sub = scenarios_df[scenarios_df["method"] == label].set_index("scenario")
        ax.bar(x + i * width, [sub.loc[s, metric] for s in scenarios],
               width=width, color=colors[col], label=label, alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([s.split(":")[0] for s in scenarios])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
scenarios = scenarios_df["scenario"].unique()
x = np.arange(len(scenarios))
for i, (col, label) in enumerate(methods.items()):
    sub = scenarios_df[scenarios_df["method"] == label].set_index("scenario")
    ratios = [sub.loc[s, "pct_dr_captured"] / sub.loc[s, "pct_days_selected"] for s in scenarios]
    ax.bar(x + i * 0.35, ratios, width=0.35, color=colors[col], label=label, alpha=0.85)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (ratio=1)")
ax.set_title("DR Efficiency\n(DR captured / days used)", fontsize=11)
ax.set_ylabel("Efficiency Ratio")
ax.set_xticks(x + 0.175)
ax.set_xticklabels([s.split(":")[0] for s in scenarios])
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
for col, label in methods.items():
    pts = gain_records[label]
    xs = [0] + [p[0] for p in pts]
    ys = [0] + [p[1] for p in pts]
    ax.plot(xs, ys, marker="o", color=colors[col], label=label, linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
ax.set_title("Cumulative Gain Curve\n(by cluster rank)", fontsize=11)
ax.set_xlabel("Fraction of Days Selected")
ax.set_ylabel("Fraction of DR Captured")
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax = fig.add_subplot(gs[1, 1])
for col, label in methods.items():
    means = df.groupby(col)["delta_load"].mean().sort_values(ascending=False).values
    ax.plot(range(1, len(means) + 1), means, marker="o", color=colors[col], label=label, linewidth=2)
ax.set_title("Cluster Ranking\n(mean delta_load, descending)", fontsize=11)
ax.set_xlabel("Cluster Rank")
ax.set_ylabel("Mean delta_load")
ax.set_xticks(range(1, max(
    df[col].nunique() for col in methods) + 1))
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
tbl_data = []
col_labels = ["Method", "Scenario", "% Days", "% DR"]
for _, row in scenarios_df.iterrows():
    tbl_data.append([
        row["method"].replace("DTW load-based", "DTW").replace("Euclidean load-based", "Euclid"),
        row["scenario"].split(":")[0],
        f"{row['pct_days_selected']:.1%}",
        f"{row['pct_dr_captured']:.1%}",
    ])
tbl = ax.table(cellText=tbl_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.1, 1.4)
ax.set_title("Summary Table", fontsize=11, pad=12)

plt.savefig(config.DATA_PATH + "/8_DTW_vs_Euklidian_dr_targeting_scenarios.png", dpi=150, bbox_inches="tight")
plt.show()

print(scenarios_df[["method","scenario","selected_days","total_days","pct_days_selected","pct_dr_captured"]].to_string(index=False))
# %% [markdown]
# Ανάλυση Στατιστικής Σημαντικότητας
# %%
import scikit_posthocs as sp

methods = {
    "load_cluster_id":        "DTW load-based",
    "euklidean_cluster_id":   "Euclidean load-based",
}
metrics = ["delta_load", "SEF", "PSS", "PAR"]

kw_records = []
for col, label in methods.items():
    for metric in metrics:
        groups = [grp[metric].values for _, grp in df.groupby(col)]
        stat, p = stats.kruskal(*groups)
        kw_records.append(dict(method=label, metric=metric, kruskal_stat=stat, p_value=p, significant=(p < 0.05)))

kw_df = pd.DataFrame(kw_records)
kw_df.to_csv(config.DATA_PATH + "/8_DTW_vs_Euklidian_kruskal_wallis.csv", index=False)
print("=== Kruskal-Wallis Results ===")
print(kw_df.to_string(index=False))

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
                if p_val < 0.05:
                    sig_pairs_records.append(dict(
                        method=label, metric=metric,
                        cluster_a=clusters[i], cluster_b=clusters[j],
                        dunn_p_value=p_val,
                    ))

sig_pairs_df = pd.DataFrame(sig_pairs_records)
sig_pairs_df.to_csv(config.DATA_PATH + "/8_DTW_vs_Euklidian_dunn_significant_pairs.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Kruskal-Wallis p-values per Metric — DTW vs Euclidean", fontsize=13, fontweight="bold")
for ax, (col, label) in zip(axes, methods.items()):
    sub = kw_df[kw_df["method"] == label].set_index("metric")
    p_vals = sub.loc[metrics, "p_value"].values
    bars = ax.bar(metrics, p_vals, color=["#2ca02c" if p < 0.05 else "#d62728" for p in p_vals], alpha=0.85)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="α = 0.05")
    ax.set_yscale("log")
    ax.set_title(label, fontsize=11)
    ax.set_ylabel("p-value (log scale)")
    ax.legend(fontsize=8)
    for bar, p in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{p:.2e}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(config.DATA_PATH + "/8_DTW_vs_Euklidian_kruskal_wallis.png", dpi=150, bbox_inches="tight")
plt.show()

for col, label in methods.items():
    available = [m for m in metrics if (col, m) in dunn_results]
    if not available:
        print(f"No significant KW results for {label} — skipping Dunn heatmaps.")
        continue

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"Dunn Test p-value Matrices (Holm correction) — {label}", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, available):
        dunn = dunn_results[(col, metric)]
        mat  = dunn.values.astype(float)
        n    = len(dunn.columns)
        im   = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(dunn.columns, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(dunn.index,   fontsize=8)
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
    plt.savefig(config.DATA_PATH + f"/8_DTW_vs_Euklidian_dunn_heatmaps_{safe_label}.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\n=== Statistically Significant Dunn Pairs (p < 0.05, Holm correction) ===")
if sig_pairs_df.empty:
    print("No significant pairs found.")
else:
    print(sig_pairs_df.to_string(index=False))

# %%
