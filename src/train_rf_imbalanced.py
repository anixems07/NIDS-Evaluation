
# %% Stage 0 — Locate the project + rebuild the imbalanced training set
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# --- robust project-root resolution (works from src/, root, or VS Code cells) ---
if "__file__" in globals():
    PROJECT = Path(__file__).resolve().parent.parent   # src/ -> project root
else:
    PROJECT = Path.cwd()                                # interactive cell: workspace root
    if (PROJECT / "src").exists() is False and (PROJECT.parent / "src").exists():
        PROJECT = PROJECT.parent                        # cwd inside src/ -> go up

DATA  = PROJECT / "data" / "processed"
TABS  = PROJECT / "results" / "tables"
FIGS  = PROJECT / "results" / "figures"
MODELS = PROJECT / "results" / "models"
for p in (DATA, TABS, FIGS, MODELS):
    if not p.parent.exists():
        raise FileNotFoundError(f"Missing folder structure at {p}. "
                                "Expected layout: data/processed, results/tables, ...")

clean_csv = DATA / "cicids2017_clean.csv"
if not clean_csv.exists():
    raise FileNotFoundError(f"Could not find {clean_csv}. Run preprocess_v2.py first.")

print("Project root :", PROJECT)
print("Data folder  :", DATA)

# --- rebuild the natural-ratio training set (identical rows to your original split) ---
df = pd.read_csv(clean_csv)
known = df[~df["Label"].isin(["Heartbleed", "Infiltration"])]   # same zero-day holdout
X = known.drop(columns=["Label", "Label_encoded"])
y = known["Label_encoded"]

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

name_to_code = dict(zip(df["Label"], df["Label_encoded"]))
benign_code = name_to_code["BENIGN"]

# --- sanity checks: PROOF this is the same split your SMOTE set came from ---
print("Imbalanced training set :", X_tr.shape)                       # (1786493, 46)
print("Test set                :", X_te.shape)                       # (446624, 46)
print("BENIGN share of training:", round((y_tr == benign_code).mean() * 100, 1), "%")  # ~84.9
print("Bot rows in training    :", (y_tr == name_to_code["Bot"]).sum())                # ~1150

saved_test = DATA / "X_test.csv"
if saved_test.exists():
    print("Same exam as before    :",
          X_te.reset_index(drop=True).equals(pd.read_csv(saved_test)))


# %% Stage 1 — Train: identical settings to the SMOTE run (only the data differs)
from sklearn.ensemble import RandomForestClassifier

model_imb = RandomForestClassifier(
    n_estimators=100,   # SAME as SMOTE run — this equality IS the experiment
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
model_imb.fit(X_tr, y_tr)
print(f"Training finished in {(time.time()-t0)/60:.1f} minutes")


# %% Stage 2 — Evaluate on the SAME untouched test set + full metric set
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, balanced_accuracy_score,
                             matthews_corrcoef)

t0 = time.time()
y_pred = model_imb.predict(X_te)
print(f"446,624 predictions in {time.time()-t0:.1f} seconds")

mapping = pd.read_csv(TABS / "label_mapping.csv")
code_to_name = dict(zip(mapping["code"], mapping["label_name"]))
labels_in_use = sorted(y_te.unique())
names = [code_to_name[c] for c in labels_in_use]

report = classification_report(y_te, y_pred, labels=labels_in_use,
                               target_names=names, digits=3)
print(report)

acc  = accuracy_score(y_te, y_pred)
mf1  = f1_score(y_te, y_pred, average="macro")
wf1  = f1_score(y_te, y_pred, average="weighted")
bacc = balanced_accuracy_score(y_te, y_pred)
mcc  = matthews_corrcoef(y_te, y_pred)
benign_fpr = (y_pred[(y_te == benign_code).values] != benign_code).mean()

print(f"Accuracy           : {acc:.4f}   (SMOTE run: 0.9983)")
print(f"Macro F1           : {mf1:.4f}   (SMOTE run: 0.9495)")
print(f"Weighted F1        : {wf1:.4f}   (SMOTE run: 0.9984)")
print(f"Balanced Accuracy  : {bacc:.4f}   (SMOTE run: 0.9858)")
print(f"MCC                : {mcc:.4f}   (SMOTE run: 0.9937)")
print(f"Benign FPR         : {benign_fpr:.4%}  (SMOTE run: 0.1421%)")

(TABS / "imb_classification_report.txt").write_text(report)
(TABS / "imb_metrics_summary.txt").write_text(
    f"Accuracy: {acc:.4f}\nMacro F1: {mf1:.4f}\nWeighted F1: {wf1:.4f}\n"
    f"Balanced Accuracy: {bacc:.4f}\nMCC: {mcc:.4f}\nBenign FPR: {benign_fpr:.4%}")


# %% Stage 3 — Confusion matrix (table + visual, same format as the SMOTE run)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm_raw = confusion_matrix(y_te, y_pred, labels=labels_in_use)
pd.DataFrame(cm_raw, index=names, columns=names).to_csv(TABS / "imb_confusion_matrix.csv")

cm_norm = confusion_matrix(y_te, y_pred, labels=labels_in_use, normalize="true")
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay(cm_norm, display_labels=names).plot(
    ax=ax, cmap="Blues", values_format=".2f")
ax.set_title("Random Forest (imbalanced / no SMOTE)\nRow-normalized — diagonal = per-class recall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGS / "imb_confusion_matrix.png", dpi=150)
plt.show()


# %% Stage 4 — Persist the model (cross-dataset + SHAP stages will load this)
joblib.dump(model_imb, MODELS / "rf_model_imbalanced.joblib")
print("Saved:", MODELS / "rf_model_imbalanced.joblib")


# %% Stage 5 — Zero-day probe: the 47 never-seen attacks, this model's verdict
X_zd = pd.read_csv(DATA / "X_zero_day.csv")
zd_pred = pd.Series(model_imb.predict(X_zd)).map(code_to_name)
counts = zd_pred.value_counts()
print(counts)
(TABS / "imb_zero_day_results.txt").write_text(
    "Zero-day probe — RF (imbalanced regime)\n\n" + counts.to_string())

# %% Stage 6 — THE VERDICT (file-based): SMOTE numbers read from result files,
#              imbalanced numbers from this session's run — printed AND saved.
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# --- locate the SMOTE result files (accepts new AND old names) ---
def find_file(*names):
    for n in names:
        p = TABS / n
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of {list(names)} found in {TABS}.\n"
        "Locate it with:  find . -name '*classification_report*'\n"
        "then rename classification_report.txt -> rf_classification_report.txt"
    )

metrics_path = find_file("rf_metrics_summary.txt", "metrics_summary.txt")
report_path  = find_file("rf_classification_report.txt", "classification_report.txt")
print("SMOTE numbers read from:", metrics_path.name, "+", report_path.name)

# --- parse the aggregate metrics file (lines like "Accuracy: 0.9983") ---
smote_vals = {}
for line in metrics_path.read_text().splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        smote_vals[k.strip()] = v.strip()

# --- parse the per-class report (handles class names that contain spaces) ---
def parse_report(path):
    rows = {}
    for line in path.read_text().splitlines():
        parts = line.strip().rsplit(None, 4)   # split from the RIGHT -> class name stays whole
        if len(parts) != 5:
            continue                            # header / blank / accuracy lines
        if parts[0] == "accuracy" or parts[0].endswith("avg"):
            continue                            # summary rows, not classes
        try:
            rows[parts[0]] = {"precision": float(parts[1]), "recall": float(parts[2]),
                              "f1": float(parts[3]), "support": int(parts[4])}
        except ValueError:
            continue
    return rows

smote_rep = parse_report(report_path)
print("Per-class rows parsed from report:", len(smote_rep))

# --- 1) aggregate verdict table ---
summary = pd.DataFrame({
    "metric":     ["Accuracy", "Macro F1", "Weighted F1", "Balanced Accuracy", "MCC", "Benign FPR"],
    "smote":      [smote_vals.get("Accuracy"),  smote_vals.get("Macro F1"),
                   smote_vals.get("Weighted F1"), smote_vals.get("Balanced Accuracy"),
                   smote_vals.get("MCC"), smote_vals.get("Benign FPR")],
    "imbalanced": [f"{acc:.4f}", f"{mf1:.4f}", f"{wf1:.4f}",
                   f"{bacc:.4f}", f"{mcc:.4f}", f"{benign_fpr:.4%}"],
})
summary.to_csv(TABS / "smote_vs_imbalanced_summary.csv", index=False)
print(summary.to_string(index=False))
if summary["smote"].isna().any():
    print("WARNING: some SMOTE metrics missing from", metrics_path.name, "— check its key names")

# --- 2) per-class verdict table (imbalanced live, SMOTE from file) + deltas ---
rows = []
for c in labels_in_use:
    name = code_to_name[c]
    y_true_c = (y_te.values == c)
    sm = smote_rep.get(name, {})
    r = {
        "class":   name,
        "support": int(y_true_c.sum()),
        "imb_precision":    precision_score(y_true_c, (y_pred == c), zero_division=0),
        "imb_recall":       recall_score(y_true_c, (y_pred == c), zero_division=0),
        "imb_f1":           f1_score(y_true_c, (y_pred == c), zero_division=0),
        "smote_precision":  sm.get("precision"),
        "smote_recall":     sm.get("recall"),
        "smote_f1":         sm.get("f1"),
        "recall_delta":     (sm.get("recall", 0) - recall_score(y_true_c, (y_pred == c), zero_division=0)) if sm else None,
        "f1_delta":         (sm.get("f1", 0) - f1_score(y_true_c, (y_pred == c), zero_division=0)) if sm else None,
    }
    rows.append(r)

per_class = pd.DataFrame(rows).sort_values("support")
per_class.to_csv(TABS / "smote_vs_imbalanced_per_class.csv", index=False)
print("\n", per_class.round(3).to_string(index=False))
print("\nSaved:", TABS / "smote_vs_imbalanced_summary.csv")
print("Saved:", TABS / "smote_vs_imbalanced_per_class.csv")

missing = [code_to_name[c] for c in labels_in_use if code_to_name[c] not in smote_rep]
if missing:
    print("WARNING — classes not found in the SMOTE report file:", missing)
# %%
