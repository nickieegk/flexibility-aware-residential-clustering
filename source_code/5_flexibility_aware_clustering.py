# %% [markdown]
# # 5. Ομαδοποίηση με Βάση την Ευελιξία
#
# Αυτό το notebook διαβάζει το `4_days_metrics.csv`, δηλαδή τον πίνακα metrics
# ανά κατοικία-ημέρα που δημιουργήθηκε μετά το load clustering και τον
# υπολογισμό ευελιξίας.
#
# Χρησιμοποιεί τα ημερήσια χαρακτηριστικά ευελιξίας και demand response `SEF`,
# `PSS`, `PAR` και `Slack_i`, τα κανονικοποιεί και εκτελεί k-means με `k=8`.
# Σε αντίθεση με το προηγούμενο load-profile clustering, εδώ οι ημέρες
# ομαδοποιούνται με βάση τη συμπεριφορά ευελιξίας και τη δυνατότητα μετατόπισης
# φορτίου στην αιχμή, όχι με βάση ολόκληρο το 96-σημείων σχήμα κατανάλωσης.
#
# Το αποτέλεσμα είναι το `5_days_metrics_CLUSTERED_both_ways.csv`: ο πίνακας
# metrics εμπλουτισμένος με `flexibility_cluster_id`, διατηρώντας παράλληλα το
# προηγούμενο `load_cluster_id`, ώστε τα επόμενα notebooks να συγκρίνουν τις δύο
# προσεγγίσεις clustering.
# %%
import sys
import os
sys.path.append(os.path.abspath('../'))
import config
from helpers import (
    load_flexibility_metrics,
    run_flexibility_kmeans_clustering,
    save_flexibility_clusters,
    scale_flexibility_features,
)

K              = 8
MAX_ITER_FINAL = 300
N_INIT_FINAL   = 5
RANDOM_STATE   = 42

FEATURE_COLS = ["SEF", "PSS", "PAR", "Slack_i"]
META_COLS    = ["house_id", "date"]

INPUT_CSV  = config.OUTPUT_PATH + "/4_days_metrics.csv"
OUTPUT_CSV = config.DATA_PATH  + "/5_days_metrics_CLUSTERED_both_ways.csv"


X_raw, df_raw = load_flexibility_metrics(
    INPUT_CSV,
    feature_cols=FEATURE_COLS,
    meta_cols=META_COLS,
)

X_scaled = scale_flexibility_features(X_raw)

labels, _model = run_flexibility_kmeans_clustering(
    X_scaled     = X_scaled,
    k            = K,
    max_iter     = MAX_ITER_FINAL,
    n_init       = N_INIT_FINAL,
    random_state = RANDOM_STATE,
)

df_out = save_flexibility_clusters(df_raw, labels, OUTPUT_CSV)

print(f"Αποτελέσματα → '{OUTPUT_CSV}'")
print(f"\nΚατανομή flexibility_cluster_id:")
print(df_out["flexibility_cluster_id"].value_counts().sort_index().to_string())
# %%
