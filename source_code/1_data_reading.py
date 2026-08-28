# %% [markdown]
# # 1. Ανάγνωση και Ενοποίηση Δεδομένων
#
# Αυτό το notebook διαβάζει τους αρχικούς φακέλους κατοικιών μέσα από το
# `config.DATA_PATH`, όπου κάθε φάκελος `House_<id>` περιέχει ηλεκτρικές
# και περιβαλλοντικές μετρήσεις. Τα ηλεκτρικά CSV περιέχουν timestamped 
# στήλες συνολικής κατανάλωσης και κατανάλωσης ανά συσκευή, ενώ τα
# περιβαλλοντικά CSV περιέχουν αντίστοιχες μετρήσεις θερμοκρασίας και υγρασίας.
#
# Καθαρίζει βασικά προβλήματα των raw δεδομένων αφαιρώντας στήλες τάσης/έντασης,
# γραμμές με `issues`, μετατρέποντας τις ηλεκτρικές μετρήσεις σε διαστήματα
# 15 λεπτών και ενώνοντας τα ηλεκτρικά με τα περιβαλλοντικά δεδομένα κάθε
# κατοικίας με βάση το `timestamp`. Επίσης διορθώνει τη διπλή ονομασία της
# στήλης εξωτερικής θερμοκρασίας.
#
# Το αποτέλεσμα είναι το `1_all_data_merged.csv`: ένας ταξινομημένος πίνακας με
# `house_id`, `timestamp`, στήλες συνολικής/ανά συσκευή κατανάλωσης και στήλες
# εσωτερικών/εξωτερικών περιβαλλοντικών μετρήσεων για όλες τις κατοικίες.
# %%
import sys
import os
sys.path.append(os.path.abspath('..'))
import config

import pandas as pd
from pathlib import Path
import re

data_path = Path(config.DATA_PATH)

def load_init_data(house_id: str, inner_folder: str) -> pd.DataFrame:
    folder = data_path / f"House_{house_id}" / inner_folder
    files = sorted(folder.glob("*.csv"))
    dfs = []
    
    for f in files:
        if "metadata" in f.name.lower():
            continue

        df = pd.read_csv(f)
        
        df = df.drop(columns=['V', 'A'], errors='ignore')
        
        if 'issues' in df.columns:
            df = df[df['issues'] == 0].drop(columns=['issues'])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def custom_mean(x: pd.Series) -> float:
    non_null = x.dropna()
    
    if non_null.empty:
        return 0.0

    if x.name == 'P_agg':
        non_zero = non_null[non_null != 0]
        return non_zero.mean() if not non_zero.empty else 0.0
    else:
        return non_null.mean()

pattern = re.compile(r"^House_(\d+)$")
houses_dfs = []

for folder in data_path.iterdir():
    if folder.is_dir():
        match = pattern.match(folder.name)
        if match:
            house_id = match.group(1)
            
            el_df = load_init_data(house_id, "Electric_data")
            
            el_df = (
                el_df.set_index("timestamp")
                .resample("15min")
                .apply(custom_mean)
                .reset_index()
            )

            env_df = load_init_data(house_id, "Environmental_data")

            merged_df = el_df.merge(env_df, on="timestamp", how="left")
            merged_df.insert(0, "house_id", house_id)

            print(f"House {house_id}: {merged_df.shape}")
            houses_dfs.append(merged_df)

full_df = pd.concat(houses_dfs, ignore_index=True)
full_df.sort_values(by=["house_id", "timestamp"], inplace=True)
full_df.reset_index(drop=True, inplace=True)

mask_both = (full_df["external_temperature"].notna() & 
             full_df["external_temparature"].notna())

if mask_both.any():
    raise ValueError(
        f"Βρέθηκαν {mask_both.sum()} γραμμές με τιμές και στις δύο στήλες θερμοκρασίας"
    )

mask_copy = (full_df["external_temperature"].isna() & 
             full_df["external_temparature"].notna())
full_df.loc[mask_copy, "external_temperature"] = full_df.loc[mask_copy, "external_temparature"]

full_df = full_df.drop(columns=["external_temparature"])

output_full_path = config.DATA_PATH + "/1_all_data_merged.csv";

full_df.to_csv(
    output_full_path, 
    index=False
)

print(f"\nΣυνολικό shape: {full_df.shape}\n")
print(full_df.head())
print(f"\nΤο τελικό dataframe αποθηκεύτηκε επιτυχώς στο '{output_full_path}'")
# %%
