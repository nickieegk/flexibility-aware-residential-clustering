# %% [markdown]
# # 4. Ημερήσια Metrics Ευελιξίας
#
# Αυτό το notebook διαβάζει το συμπληρωμένο dataset σε επίπεδο 15λεπτου
# `2_targeted_period_imputated.csv` και τα load-cluster assignments από το
# `3_daily_profiles_CLUSTERED.csv`. Τα δεδομένα 15λεπτου περιλαμβάνουν συνολική
# κατανάλωση, κατανάλωση ανά συσκευή, timestamps και αναγνωριστικά κατοικιών.
#
# Ενώνει τα κανάλια των air conditioners, υπολογίζει ημερήσια ενέργεια
# μετατοπίσιμων συσκευών, συνολική ενέργεια, ενέργεια στην περίοδο αιχμής και
# peak-to-average συμπεριφορά. Έπειτα υπολογίζει household-level flexibility
# slack με βάση το πόσο σταθερά ξεκινούν οι μετατοπίσιμες συσκευές μέσα στη
# μέρα. Αυτές οι τιμές χρησιμοποιούνται για την εκτίμηση του `delta_load`, δηλαδή
# του ευέλικτου φορτίου που είναι διαθέσιμο στο βραδινό peak window.
#
# Τα αποτελέσματα είναι τα:
# 1.    `4_slack_per_house.csv`, με το slack και τον συντελεστή theta κάθε κατοικίας, και το 
# 2.    `4_days_metrics.csv`, με μία γραμμή ανά κατοικία-ημέρα που περιέχει load cluster, SEF, PSS, PAR, ενεργειακά σύνολα, slack τιμές και `delta_load`.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config
import pandas as pd
import numpy as np

SHIFTABLE_APPLIANCES = ["air_condition", "boiler", "washing_machine", "dishwasher"]

STANDBY_THRESHOLDS = {
    "air_condition":   50.0,
    "boiler":         100.0,
    "washing_machine": 20.0,
    "dishwasher":      10.0,
}

PEAK_WINDOW_START  = "18:00"
PEAK_WINDOW_END    = "20:45"
MIN_EVENT_INTERVALS = 2   # ≥ 30 min of consecutive intervals to count as an event

THETA_BASE  = 0.1
THETA_SCALE = 0.3

def combine_ac_columns(df: pd.DataFrame) -> pd.DataFrame:
    ac_cols = ["ac_1", "ac_2", "ac_3"]
    existing = [c for c in ac_cols if c in df.columns]
    missing  = [c for c in ac_cols if c not in df.columns]
    if missing:
        print(f"Warning: AC columns missing, treated as zero: {missing}")
    df["air_condition"] = df[existing].sum(axis=1) if existing else 0.0
    return df


def detect_event_starts(series: pd.Series, threshold: float,
                         min_intervals: int = MIN_EVENT_INTERVALS) -> list[int]:
    is_on      = series.values > threshold
    timestamps = series.index
    starts, i, n = [], 0, len(is_on)

    while i < n:
        if is_on[i]:
            run_start = i
            while i < n and is_on[i]:
                i += 1
            if (i - run_start) >= min_intervals:
                ts = timestamps[run_start]
                starts.append(int(ts.hour * 60 + ts.minute))
        else:
            i += 1
    return starts


def circular_slack(start_times: list[int], period: int = 1440) -> float:
    if len(start_times) < 2:
        return 0.0
    angles = 2 * np.pi * np.array(start_times) / period
    R = np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2)
    return float(1.0 - R)


def compute_user_slack(house_df: pd.DataFrame) -> dict:
    house_df = house_df.sort_values("timestamp")
    slacks, energies = [], []

    for appliance in SHIFTABLE_APPLIANCES:
        if appliance not in house_df.columns:
            slacks.append(0.0)
            energies.append(0.0)
            continue
        series   = house_df.set_index("timestamp")[appliance]
        threshold = STANDBY_THRESHOLDS.get(appliance, 0.0)
        slacks.append(circular_slack(detect_event_starts(series, threshold)))
        energies.append(house_df[appliance].sum())

    slacks   = np.array(slacks)
    energies = np.array(energies)
    total    = energies.sum()

    slack_i = float(np.dot(slacks, energies) / total) if total > 0 else 0.0
    theta_i = THETA_BASE + THETA_SCALE * slack_i

    return {
        "Slack_i": round(slack_i, 6),
        "theta_i": round(theta_i, 6),
        **{f"slack_{a}": round(s, 6) for a, s in zip(SHIFTABLE_APPLIANCES, slacks)},
    }

df = pd.read_csv(config.DATA_PATH + "/2_targeted_period_imputated.csv",
                 parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
df["date"]      = pd.to_datetime(df["timestamp"].dt.date)
df["time"]      = df["timestamp"].dt.time

df = combine_ac_columns(df)

# all shiftable columns are now guaranteed present
for col in SHIFTABLE_APPLIANCES:
    if col not in df.columns:
        df[col] = 0.0

df["shiftable_total"] = df[SHIFTABLE_APPLIANCES].sum(axis=1)

daily_groups = df.groupby(["house_id", "date"])

metrics = pd.DataFrame({
    "daily_shiftable_energy": daily_groups["shiftable_total"].sum(),
    "daily_pagg_energy":      daily_groups["P_agg"].sum(),
    "daily_max_pagg":         daily_groups["P_agg"].max(),
    "daily_mean_pagg":        daily_groups["P_agg"].mean(),
}).reset_index()

metrics["SEF"] = np.where(metrics["daily_pagg_energy"] != 0,
                           metrics["daily_shiftable_energy"] / metrics["daily_pagg_energy"],
                           np.nan)
metrics["PAR"] = np.where(metrics["daily_mean_pagg"] != 0,
                           metrics["daily_max_pagg"] / metrics["daily_mean_pagg"],
                           np.nan)

peak_start = pd.to_datetime(PEAK_WINDOW_START).time()
peak_end   = pd.to_datetime(PEAK_WINDOW_END).time()
peak_mask  = (df["time"] >= peak_start) & (df["time"] <= peak_end)

peak_groups = df[peak_mask].groupby(["house_id", "date"])
peak_df = pd.DataFrame({
    "peak_shiftable_energy": peak_groups["shiftable_total"].sum(),
    "peak_pagg_energy":      peak_groups["P_agg"].sum(),
}).reset_index()

metrics = metrics.merge(peak_df, on=["house_id", "date"], how="left")
metrics[["peak_shiftable_energy", "peak_pagg_energy"]] = \
    metrics[["peak_shiftable_energy", "peak_pagg_energy"]].fillna(0)

metrics["PSS"] = np.where(metrics["peak_pagg_energy"] != 0,
                           metrics["peak_shiftable_energy"] / metrics["peak_pagg_energy"],
                           np.nan)

clusters = pd.read_csv(config.DATA_PATH + "/3_daily_profiles_CLUSTERED.csv",
                       usecols=["house_id", "date", "load_cluster_id"])
clusters["date"] = pd.to_datetime(clusters["date"])

final_df = metrics.merge(clusters, on=["house_id", "date"], how="left")

unmatched = final_df["load_cluster_id"].isna().sum()
if unmatched:
    print(f"Warning: {unmatched} day(s) could not be matched to a load_cluster_id.")

slack_records = []
for house_id in sorted(df["house_id"].unique()):
    house_df = df[df["house_id"] == house_id][["house_id", "timestamp"] + SHIFTABLE_APPLIANCES]
    result   = compute_user_slack(house_df)
    slack_records.append({"house_id": house_id, **result})

slack_df = pd.DataFrame(slack_records).sort_values("house_id").reset_index(drop=True)

output_full_path = config.OUTPUT_PATH + "/4_slack_per_house.csv";

slack_df[["house_id", "Slack_i", "theta_i"]].to_csv(output_full_path, index=False)
print(f"Το slack-dataframe αποθηκεύτηκε επιτυχώς στο '{output_full_path}'\n")

final_df = final_df.merge(
    slack_df[["house_id", "Slack_i", "theta_i"]], 
    on="house_id", how="left"
)

final_df["delta_load"] = final_df["theta_i"] * final_df["peak_shiftable_energy"]

output_columns = [
    "house_id", "date", "load_cluster_id", "SEF", "PSS", "PAR",
    "daily_shiftable_energy", "daily_pagg_energy", "peak_shiftable_energy", 
    "peak_pagg_energy", "daily_max_pagg", "daily_mean_pagg", "Slack_i", 
    "theta_i", "delta_load"
]

output_full_path = config.OUTPUT_PATH + "/4_days_metrics.csv";

final_df = final_df[output_columns].sort_values(["house_id", "date"]).reset_index(drop=True)
final_df.to_csv(output_full_path, index=False)
print(f"Το dataframe όλων των metrics αποθηκεύτηκε επιτυχώς στο '{output_full_path}'")
# %%
