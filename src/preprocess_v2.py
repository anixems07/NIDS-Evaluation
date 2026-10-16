"""
CICIDS2017 — Preprocessing v2
Goals: clean data, honest eval, imbalance fix, zero-day experiment setup, low-FP design
Expects raw CSVs in data/raw/
Outputs processed datasets to data/processed/ and metrics/tables to results/tables/
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ---------------- paths & settings ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "results" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CORR_THRESHOLD = 0.95

BENIGN_TARGET    = 300_000                    # normal rows kept in train (FP guard)
ATTACK_CAP       = 50_000                     # max rows per attack class in train
ZERO_DAY_CLASSES = ["Heartbleed", "Infiltration"]  # hidden from training = simulated zero-days

summary_lines = []
def log(msg):
    print(msg)
    summary_lines.append(msg)

# ---------------------------------------------------------------------------
# Step 1: Load and merge the 8 CSVs
# ---------------------------------------------------------------------------
files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in '{RAW_DIR}'.")

log(f"Found {len(files)} CSV files")
dfs = [pd.read_csv(f, encoding="latin1", low_memory=False) for f in files]
df = pd.concat(dfs, ignore_index=True)
del dfs                                    # CHANGED: frees ~2-3 GB immediately
raw_shape = df.shape
log(f"Raw merged shape: {raw_shape}")

# Step 2: strip column names (unchanged)
df.columns = [c.strip() for c in df.columns]

# Step 3: drop leakage columns (unchanged)
leak_cols = ["Source IP", "Destination IP", "Source Port", "Destination Port",
             "Timestamp", "Flow ID"]
present = [c for c in leak_cols if c in df.columns]
df = df.drop(columns=present)
log(f"Dropped leakage columns: {present}")

# ---------------------------------------------------------------------------
# Step 4: Fix infinite/missing values
# CHANGED: 'Infinity' arrives as TEXT, so select_dtypes misses it -> SMOTE crashes.
# Convert text feature columns to numeric FIRST.
# ---------------------------------------------------------------------------
obj_feature_cols = [c for c in df.columns if df[c].dtype == object and c != "Label"]
for c in obj_feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if obj_feature_cols:
    log(f"Converted text feature columns to numeric: {obj_feature_cols}")

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
rows_before_dropna = df.shape[0]
df = df.dropna()
rows_after_dropna = df.shape[0]
log(f"Rows dropped for inf/NaN: {rows_before_dropna - rows_after_dropna}")

# Step 5: dedup (unchanged)
rows_before_dedup = df.shape[0]
df = df.drop_duplicates()
rows_after_dedup = df.shape[0]
log(f"Duplicates removed: {rows_before_dedup - rows_after_dedup}")

# Step 6: zero-variance columns (unchanged)
feature_cols = [c for c in df.columns if c not in ("Label",)]
nunique = df[feature_cols].nunique()
zero_var_cols = nunique[nunique <= 1].index.tolist()
df = df.drop(columns=zero_var_cols)
log(f"Dropped zero-variance columns: {zero_var_cols}")

# ---------------------------------------------------------------------------
# Step 7 (NEW): merge the 3 "Web Attack" sub-labels into one class
# ---------------------------------------------------------------------------
web_mask = df["Label"].str.startswith("Web Attack")
log(f"Merging web subtypes: {df.loc[web_mask, 'Label'].value_counts().to_dict()}")
df.loc[web_mask, "Label"] = "Web Attack"

# class distribution before balancing (unchanged, now 13 classes)
class_counts = df["Label"].value_counts()
class_counts.to_csv(TABLES_DIR / "class_distribution_before_balancing.csv")
log(f"\nClass distribution before balancing:\n{class_counts.to_string()}\n")

# Step 8: correlation pruning (unchanged)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop_corr = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
df = df.drop(columns=to_drop_corr)
log(f"Dropped {len(to_drop_corr)} highly correlated columns")

# ---------------------------------------------------------------------------
# Step 9: encode labels  (CHANGED: mapping now includes attack family)
# ---------------------------------------------------------------------------
le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Label"])
FAMILY = {
    "BENIGN": "Benign", "Bot": "Bot", "DDoS": "DDoS",
    "DoS GoldenEye": "DoS", "DoS Hulk": "DoS",
    "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
    "Heartbleed": "Exploit", "Infiltration": "Infiltration",
    "PortScan": "Recon", "Web Attack": "Web",
}
mapping = pd.DataFrame({
    "code": range(len(le.classes_)),
    "label_name": le.classes_,
    "family": [FAMILY.get(c, "Other") for c in le.classes_],
})
mapping.to_csv(TABLES_DIR / "label_mapping.csv", index=False)
mapping.to_csv(PROCESSED_DIR / "label_mapping.csv", index=False)
log(f"\nLabel mapping:\n{mapping.to_string(index=False)}")

# Step 10: save cleaned dataset (unchanged)
clean_shape = df.shape
df.to_csv(PROCESSED_DIR / "cicids2017_clean.csv", index=False)
log(f"Saved cleaned dataset, shape={clean_shape}")

# ---------------------------------------------------------------------------
# Step 11 (CHANGED): zero-day holdout + stratified split of KNOWN classes only
# ---------------------------------------------------------------------------
zero_mask = df["Label"].isin(ZERO_DAY_CLASSES)
known_df = df.loc[~zero_mask].copy()
zd_df    = df.loc[zero_mask].copy()
log(f"\nZERO-DAY holdout {ZERO_DAY_CLASSES}: {len(zd_df)} rows (never trained on, never SMOTEd)")
log(f"Known classes for training: {known_df['Label'].nunique()}")

X = known_df.drop(columns=["Label", "Label_encoded"])
y = known_df["Label_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_zero = zd_df.drop(columns=["Label", "Label_encoded"])
y_zero = pd.Series(zd_df["Label_encoded"].values, name="label")
log(f"Split: train={X_train.shape}, test={X_test.shape}, zero-day probe={X_zero.shape}")

pd.Series(X_train.columns).to_csv(TABLES_DIR / "final_feature_list.csv",
                                  index=False, header=["feature"])

# ---------------------------------------------------------------------------
# Step 12 (NEW): benign-only sample + scaling stats -> for the anomaly detector
# ---------------------------------------------------------------------------
benign_code = int(le.transform(["BENIGN"])[0])
benign_train = X_train.loc[y_train == benign_code]
if len(benign_train) > BENIGN_TARGET:
    benign_train = benign_train.sample(n=BENIGN_TARGET, random_state=42)
benign_train.to_csv(PROCESSED_DIR / "benign_train.csv", index=False)

scaler_stats = pd.DataFrame({
    "feature": X_train.columns,
    "mean": benign_train.mean().values,
    "std":  benign_train.std().values,
})
scaler_stats["std"] = scaler_stats["std"].replace(0, 1)   # avoid divide-by-zero
scaler_stats.to_csv(PROCESSED_DIR / "scaler_stats.csv", index=False)
log(f"Saved benign-only sample: {benign_train.shape} (+ scaler stats)")

# ---------------------------------------------------------------------------
# Step 13 (CHANGED): hybrid balancing — undersample big classes, SMOTE small ones
# ---------------------------------------------------------------------------
counts = y_train.value_counts()
under_targets, over_targets = {}, {}
for code, n in counts.items():
    if code == benign_code:
        if n > BENIGN_TARGET:
            under_targets[code] = BENIGN_TARGET
    else:
        if n > ATTACK_CAP:
            under_targets[code] = ATTACK_CAP
        elif n < ATTACK_CAP:
            if n < 6:
                log(f"WARNING: class code {code} has {n} rows - left natural (too few to synthesize)")
            else:
                over_targets[code] = ATTACK_CAP

X_u, y_u = X_train, y_train
if under_targets:
    X_u, y_u = RandomUnderSampler(sampling_strategy=under_targets,
                                   random_state=42).fit_resample(X_train, y_train)
X_train_bal, y_train_bal = X_u, y_u
if over_targets:
    X_train_bal, y_train_bal = SMOTE(sampling_strategy=over_targets,
                                     random_state=42, k_neighbors=5).fit_resample(X_u, y_u)

after_counts = pd.Series(y_train_bal).value_counts()
after_counts.to_csv(TABLES_DIR / "class_distribution_after_balancing.csv")
log(f"\nTraining set after balancing: {X_train_bal.shape}\n{after_counts.to_string()}")

# ---------------------------------------------------------------------------
# Save splits (X/y names unchanged so your existing training code still works)
# ---------------------------------------------------------------------------
pd.DataFrame(X_train_bal, columns=X_train.columns).to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
pd.Series(y_train_bal, name="label").to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.rename("label").to_csv(PROCESSED_DIR / "y_test.csv", index=False)           # header preserved
X_zero.to_csv(PROCESSED_DIR / "X_zero_day.csv", index=False)
y_zero.to_csv(PROCESSED_DIR / "y_zero_day.csv", index=False)

log(f"""
=========================================
PREPROCESSING SUMMARY (v2)
=========================================
Raw rows:                  {raw_shape[0]}
After inf/NaN removal:     {rows_after_dropna}
After dedup:               {rows_after_dedup}
Final feature count:       {X_train.shape[1]}
Classes in train (known):  {y.nunique()}
Zero-day rows held out:    {len(zd_df)}
Train (balanced):          {X_train_bal.shape}
Test (untouched, real ratio): {X_test.shape}
Benign sample (anomaly):   {benign_train.shape}
========================================""")

with open(TABLES_DIR / "preprocessing_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\nDone.")