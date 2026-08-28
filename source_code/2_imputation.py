# %% [markdown]
# # 2. Συμπλήρωση Κενών Τιμών και Ημερήσια Προφίλ
#
# Αυτό το notebook διαβάζει το `1_all_data_merged.csv`, δηλαδή την ενιαία
# χρονοσειρά ανά 15 λεπτά για κάθε κατοικία, με συνολική κατανάλωση,
# κατανάλωση ανά συσκευή και περιβαλλοντικές μετρήσεις.
#
# Περιορίζει τα δεδομένα στην περίοδο ανάλυσης, συμπληρώνει κενές τιμές
# εξωτερικής θερμοκρασίας και υγρασίας χρησιμοποιώντας πληροφορία από άλλες
# κατοικίες στο ίδιο `timestamp`, αφαιρεί τις κατοικίες 2 και 8, θεωρεί τη
# μηδενική συνολική κατανάλωση ως κενή τιμή και συμπληρώνει τα υπόλοιπα κενά.
# Τα μικρά κενά συμπληρώνονται από γειτονικές τιμές της ίδιας κατοικίας, ενώ τα
# μεγαλύτερα από κοντινές παρατηρήσεις ίδιας ημέρας εβδομάδας και ίδιας ώρας.
#
# Τα αποτελέσματα είναι τα:
# 1.    `2_targeted_period_imputated.csv`, ένα καθαρισμένο dataset σε επίπεδο 15λεπτου, και το
# 2.    `2_targeted_period_imputated_daily_profiles.csv`, όπου κάθε γραμμή είναι ένα ζεύγος κατοικίας-ημέρας και οι 96 τιμές 15λεπτου του `P_agg` έχουν απλωθεί σε στήλες ώρας για χρήση στο clustering.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config

import pandas as pd
import numpy as np

df = pd.read_csv(
    config.DATA_PATH + "/1_all_data_merged.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values(["house_id", "timestamp"])
print("Αρχικό shape:", df.shape)
df.head()

start = pd.Timestamp("2023-04-21 07:45:00")
end   = pd.Timestamp("2023-09-30 23:45:00")

range_mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)

temp_mean_by_ts = df.groupby("timestamp")["external_temperature"].transform("mean")
mask_temp_nan = df["external_temperature"].isna() & range_mask
df.loc[mask_temp_nan, "external_temperature"] = temp_mean_by_ts[mask_temp_nan]

hum_series = df.set_index(["timestamp", "house_id"])["external_humidity"]

mask_5 = (
    (df["house_id"] == 5) &
    df["external_humidity"].isna() &
    range_mask
)

if mask_5.any():
    idx_5_partner = list(zip(df.loc[mask_5, "timestamp"], [13] * mask_5.sum()))
    partner_vals_5 = hum_series.loc[idx_5_partner].to_numpy()
    df.loc[mask_5, "external_humidity"] = partner_vals_5

mask_13 = (
    (df["house_id"] == 13) &
    df["external_humidity"].isna() &
    range_mask
)

if mask_13.any():
    idx_13_partner = list(zip(df.loc[mask_13, "timestamp"], [5] * mask_13.sum()))
    partner_vals_13 = hum_series.loc[idx_13_partner].to_numpy()
    df.loc[mask_13, "external_humidity"] = partner_vals_13

mask_other_nan = (
    df["external_humidity"].isna() &
    ~df["house_id"].isin([5, 13]) &
    range_mask
)

if mask_other_nan.any():
    hum_mean_by_ts = (
        df[~df["house_id"].isin([5, 13])]
        .groupby("timestamp")["external_humidity"]
        .mean()
    )

    df.loc[mask_other_nan, "external_humidity"] = (
        df.loc[mask_other_nan, "timestamp"].map(hum_mean_by_ts)
    )

range_mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)

print(
    "Εναπομείναντα NaNs στην εξωτερική θερμοκρασία (στο διάστημα):",
    df.loc[range_mask, "external_temperature"].isna().sum()
)
print(
    "Εναπομείναντα NaNs στην εξωτερική υγρασία (στο διάστημα):",
    df.loc[range_mask, "external_humidity"].isna().sum()
)

df.sort_values(
    by=["house_id", "timestamp"],
    ascending=[True, True],
    inplace=True
)

df = df[~df["house_id"].isin([2, 8])].copy()

value_cols = [
    "P_agg",
    "internal_temperature",
    "internal_humidity",
    "external_temperature",
    "external_humidity",
]

impute_stats = {
    col: {"short_count": 0, "long_count": 0}
    for col in value_cols
}

start = "2023-04-22 00:00:00"
end   = "2023-09-30 23:45:00"

df["P_agg"] = df["P_agg"].astype(float)
df["P_agg"] = df["P_agg"].mask(df["P_agg"] == 0.0)

df = df.sort_values(["house_id", "timestamp"]).reset_index(drop=True)

def fill_short_nan_runs_group(group: pd.DataFrame,
                              cols=value_cols,
                              max_run_len: int = 4) -> pd.DataFrame:
    global impute_stats

    house_id = group.name
    group = group.sort_values("timestamp").copy()
    group["house_id"] = house_id

    for col in cols:
        if col not in group.columns:
            continue

        values = group[col].to_numpy(dtype="float64")
        is_nan = np.isnan(values)
        n = len(values)

        i = 0
        while i < n:
            if not is_nan[i]:
                i += 1
                continue

            j = i
            while j < n and is_nan[j]:
                j += 1

            run_start, run_end = i, j
            run_len = run_end - run_start

            prev_idx = run_start - 1 if run_start > 0 and not is_nan[run_start - 1] else None
            next_idx = run_end if run_end < n and not is_nan[run_end] else None

            if (
                run_len <= max_run_len
                and prev_idx is not None
                and next_idx is not None
            ):
                prev_val = values[prev_idx]
                next_val = values[next_idx]
                avg = (prev_val + next_val) / 2.0

                values[run_start:run_end] = avg
                is_nan[run_start:run_end] = False

                impute_stats[col]["short_count"] += run_len

            i = run_end

        group[col] = values

    return group

df = (
    df.sort_values(["house_id", "timestamp"])
      .groupby("house_id", group_keys=False)
      .apply(fill_short_nan_runs_group, include_groups=False)
      .reset_index(drop=True)
)

df["weekday"]     = df["timestamp"].dt.weekday
df["time_of_day"] = df["timestamp"].dt.time

def fill_long_run_row(row: pd.Series,
                      df_house: pd.DataFrame,
                      col: str):
    ts = row["timestamp"]
    w  = row["weekday"]
    t  = row["time_of_day"]

    same_slot = df_house[
        (df_house["weekday"] == w) &
        (df_house["time_of_day"] == t) &
        df_house[col].notna()
    ].sort_values("timestamp")

    prev_vals = same_slot[same_slot["timestamp"] < ts].tail(2)
    next_vals = same_slot[same_slot["timestamp"] > ts].head(2)

    neighbors = pd.concat([prev_vals, next_vals])

    if neighbors.empty:
        raise ValueError(
            f"Δεν βρέθηκαν γειτονικές τιμές για house_id={row['house_id']} στις {ts} "
            f"με την ίδια μέρα και ώρα για τη στήλη {col}"
        )

    return neighbors, neighbors[col].mean()

def fill_long_runs_for_house(group: pd.DataFrame,
                             cols=value_cols) -> pd.DataFrame:
    global impute_stats

    house_id = group.name
    group = group.sort_values("timestamp").copy()
    group["house_id"] = house_id

    for col in cols:
        if col not in group.columns:
            continue

        nan_mask = group[col].isna()
        if not nan_mask.any():
            continue

        for idx in group.loc[nan_mask].index:
            row = group.loc[idx]

            neighbors, mean_val = fill_long_run_row(row, group, col)

            group.at[idx, col] = mean_val
            impute_stats[col]["long_count"] += 1

    return group

df = (
    df.sort_values(["house_id", "timestamp"])
      .groupby("house_id", group_keys=False)
      .apply(fill_long_runs_for_house, include_groups=False)
      .reset_index(drop=True)
)

print("Στατιστικά συμπλήρωσης:", impute_stats)

df = df.drop(columns=["weekday", "time_of_day"])
df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

output_full_path = config.DATA_PATH + "/2_targeted_period_imputated.csv";

df.to_csv(
    output_full_path,
    index=False
)

print(f"\nΤο τελικό dataframe αποθηκεύτηκε επιτυχώς στο '{output_full_path}'\n")

df_daily = df.copy()

df_daily["date"] = df_daily["timestamp"].dt.date
df_daily["time"] = df_daily["timestamp"].dt.strftime("%H:%M:%S")

df_profiles = df_daily.pivot(
    index=["house_id", "date"],
    columns="time",
    values="P_agg"
).reset_index()

df_profiles.columns.name = None

df_profiles = df_profiles.sort_values(["house_id", "date"]).reset_index(drop=True)

output_full_path = config.DATA_PATH + "/2_targeted_period_imputated_daily_profiles.csv"

df_profiles.to_csv(
    output_full_path,
    index=False
)

print(f"Συνολικό shape του daily profiles dataframe: {df_profiles.shape}\n")
print(f"Το dataframe των daily profiles αποθηκεύτηκε επιτυχώς στο '{output_full_path}'")
# %%
