# %% [markdown]
# # 6. Σύγκριση Clusters Φορτίου και Ευελιξίας
#
# Αυτό το notebook διαβάζει το `5_days_metrics_CLUSTERED_both_ways.csv`, το
# οποίο περιέχει μία γραμμή ανά κατοικία-ημέρα με `load_cluster_id` και
# `flexibility_cluster_id`, καθώς και demand-response metrics όπως `delta_load`,
# `SEF`, `PSS` και `PAR`.
#
# Συγκρίνει τις δύο προσεγγίσεις clustering από διάφορες πλευρές: summary
# statistics και box plots του `delta_load`, inter-cluster variance στο δυναμικό
# demand response, targeting scenarios που ταξινομούν τα clusters με βάση το
# captured `delta_load`, και μη παραμετρικούς ελέγχους σημαντικότητας μεταξύ
# clusters.
#
# Τα αποτελέσματα είναι CSVs και σχήματα σύγκρισης για delta-load cluster
# summaries, inter-cluster variance, DR targeting scenarios, Kruskal-Wallis
# tests, σημαντικά ζεύγη Dunn και Dunn heatmaps. Αυτά τα artifacts δείχνουν αν
# τα flexibility-aware clusters διαχωρίζουν καλύτερα το demand-response
# potential σε σχέση με τα load-profile clusters.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config
from helpers import (
    run_dr_targeting_analysis,
    run_kruskal_dunn_analysis,
)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

df = pd.read_csv(config.DATA_PATH + "/5_days_metrics_CLUSTERED_both_ways.csv")

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
        .assign(method=cluster_col)
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

load_stats = cluster_stats(df, "load_cluster_id")
flex_stats = cluster_stats(df, "flexibility_cluster_id")

combined = pd.concat([load_stats, flex_stats], ignore_index=True)
combined = combined[["method", "cluster_id", "count", "mean", "median", "std", "variance", "min", "max"]]

combined.to_csv(
    config.OUTPUT_PATH + "/6_delta_load_clusters_comparison.csv",
    index=False
)

methods = {
    "load_cluster_id": load_stats,
    "flexibility_cluster_id": flex_stats
}

labels = {
    "load_cluster_id": "Load-based",
    "flexibility_cluster_id": "Flexibility-aware"
}

colors = {
    "load_cluster_id": "#4C72B0",
    "flexibility_cluster_id": "#DD8452"
}

metrics = ["mean", "median", "std"]

all_cluster_ids = sorted(
    set(load_stats["cluster_id"]).union(set(flex_stats["cluster_id"]))
)

x = np.arange(len(all_cluster_ids))
bar_width = 0.35

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "delta_load statistics per cluster — Load-based vs Flexibility-aware",
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
    config.OUTPUT_PATH + "/6_delta_load_clusters_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(combined.to_string(index=False))
# %% [markdown]
# Ανάλυση Διακύμανσης Μεταξύ Clusters
# %%
methods = {
    "load_cluster_id":        "Load-based",
    "flexibility_cluster_id": "Flexibility-aware",
}
colors = {
    "load_cluster_id":        "#4C72B0",
    "flexibility_cluster_id": "#DD8452",
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
results.to_csv(config.DATA_PATH + "/7_inter_cluster_variance.csv", index=False)

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
fig.suptitle("Inter-cluster Variance Analysis — DR Potential Separation", fontsize=14, fontweight="bold", y=0.98)
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

plt.savefig(config.DATA_PATH + "/7_inter_cluster_variance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Πείραμα Στόχευσης Demand Response
# %%
scenarios_df = run_dr_targeting_analysis(
    df                       = df,
    load_cluster_col         = "load_cluster_id",
    flexibility_cluster_col  = "flexibility_cluster_id",
    output_csv               = config.DATA_PATH + "/8_dr_targeting_scenarios.csv",
    output_png               = config.DATA_PATH + "/8_dr_targeting_scenarios.png",
)

print(scenarios_df[["method","scenario","selected_days","total_days","pct_days_selected","pct_dr_captured"]].to_string(index=False))
# %% [markdown]
# Ανάλυση Στατιστικής Σημαντικότητας
# %%
metrics = ["delta_load", "SEF", "PSS", "PAR"]

kw_df, sig_pairs_df = run_kruskal_dunn_analysis(
    df                       = df,
    load_cluster_col         = "load_cluster_id",
    flexibility_cluster_col  = "flexibility_cluster_id",
    output_kw_csv            = config.DATA_PATH + "/9_kruskal_wallis.csv",
    output_sig_pairs_csv     = config.DATA_PATH + "/9_dunn_significant_pairs.csv",
    output_kw_png            = config.DATA_PATH + "/9_kruskal_wallis.png",
    dunn_heatmap_prefix      = config.DATA_PATH + "/9_dunn_heatmaps",
    metrics                  = metrics,
)

print("=== Kruskal-Wallis Results ===")
print(kw_df.to_string(index=False))

print("\n=== Statistically Significant Dunn Pairs (p < 0.05, Holm correction) ===")
if sig_pairs_df.empty:
    print("No significant pairs found.")
else:
    print(sig_pairs_df.to_string(index=False))

# %%
