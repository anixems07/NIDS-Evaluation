# src/verify_alignment.py — final gate before any model touches the 2018 data
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
DATA, TABS = PROJECT/"data"/"processed", PROJECT/"results"/"tables"

features = pd.read_csv(TABS/"final_feature_list.csv")["feature"].tolist()
mapping  = pd.read_csv(TABS/"label_mapping.csv")
code_to_name = dict(zip(mapping["code"], mapping["label_name"]))

X18 = pd.read_csv(DATA/"X_test_2018_aligned.csv")
y18 = pd.read_csv(DATA/"y_test_2018_aligned.csv")["label"]

print("CHECK 1 — STRUCTURAL")
print("  Shape:", X18.shape, "(expect ~2.9M x 46)")
print("  Columns identical to 2017 list:", list(X18.columns) == features, "(must be True)")

print("CHECK 2 — CLEANLINESS")
print("  NaN cells:", int(X18.isna().sum().sum()), "(expect 0)")
print("  Inf cells:", int(sum(np.isinf(X18[c]).sum() for c in features)), "(expect 0)")

print("CHECK 3 — LABELS")
print("  All codes legal 2017 codes:", set(y18.unique()).issubset(set(mapping["code"])))
print("  No Heartbleed(8)/Infiltration(9):", 8 not in set(y18.unique()) and 9 not in set(y18.unique()))
print("  Classes present:", y18.nunique(), "(expect 10)")
print(y18.map(code_to_name).value_counts().to_string())

print("CHECK 4 — DISTRIBUTIONS (units/semantics per feature)")
X17 = pd.read_csv(DATA/"X_test.csv")
med = pd.DataFrame({"2017": X17[features].median(), "2018": X18[features].median()})
med["ratio"] = (med["2018"]/med["2017"]).replace([np.inf,-np.inf], np.nan)
print(med.sort_values("ratio", key=abs, ascending=False, na_position="last").round(2).head(12))

print("CHECK 5 — IDENTITY SANITY (fraction passing; want ~1.000)")
print("  Init_Win_bytes_forward in [-1, 65535]:", round(X18["Init_Win_bytes_forward"].between(-1, 65535).mean(), 4))
print("  FIN Flag Count >= 0:", round((X18["FIN Flag Count"] >= 0).mean(), 4))
print("  Total Fwd Packets integer-valued:", round((X18["Total Fwd Packets"] % 1 == 0).mean(), 4))