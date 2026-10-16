# ============================================================
# evaluate_cross_rf.py — Phase 3: RF models take the 2018 exam.
# NO retraining. Two models, one exam, one drop table.
# ============================================================

# %% Stage 0 — Load the certified cross-dataset test set
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
DATA, TABS, MODELS = PROJECT/"data"/"processed", PROJECT/"results"/"tables", PROJECT/"results"/"models"

X18 = pd.read_csv(DATA/"X_test_2018_aligned.csv")           # the frozen 2018 exam
y18 = pd.read_csv(DATA/"y_test_2018_aligned.csv")["label"]

mapping = pd.read_csv(TABS/"label_mapping.csv")
code_to_name = dict(zip(mapping["code"], mapping["label_name"]))
labels_in_use = sorted(y18.unique())                        # 10 classes
names = [code_to_name[c] for c in labels_in_use]
benign_code = int(mapping.loc[mapping.label_name == "BENIGN", "code"].iloc[0])

print("Cross-dataset test set:", X18.shape, "| classes:", len(labels_in_use))

# %% Stage 1 — Both RF models take the exam (no retraining)
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

RF_MODELS = {
    "rf_smote":      MODELS/"rf_model.joblib",           # SMOTE regime
    "rf_imbalanced": MODELS/"rf_model_imbalanced.joblib", # natural-ratio regime
}

results = {}
for tag, path in RF_MODELS.items():
    if not path.exists():
        print(f"[skip] {tag} — model not found at {path}"); continue
    model = joblib.load(path)

    t0 = time.time()
    y_pred = model.predict(X18)                           # ~2-4 min on 2.9M rows
    print(f"\n{'='*62}\n{tag} — predictions done in {(time.time()-t0)/60:.1f} min\n{'='*62}")

    rep = classification_report(y18, y_pred, labels=labels_in_use,
                               target_names=names, digits=3)
    print(rep)

    acc  = accuracy_score(y18, y_pred)
    mf1  = f1_score(y18, y_pred, average="macro")
    bacc = balanced_accuracy_score(y18, y_pred)
    mcc  = matthews_corrcoef(y18, y_pred)
    fpr  = (y_pred[(y18 == benign_code).values] != benign_code).mean()
    print(f"Accuracy {acc:.4f} | Macro F1 {mf1:.4f} | BalAcc {bacc:.4f} | "
          f"MCC {mcc:.4f} | Benign FPR {fpr:.4%}")

    (TABS / f"cross_{tag}_classification_report.txt").write_text(rep)
    (TABS / f"cross_{tag}_metrics_summary.txt").write_text(
        f"Accuracy: {acc:.4f}\nMacro F1: {mf1:.4f}\nBalanced Accuracy: {bacc:.4f}\n"
        f"MCC: {mcc:.4f}\nBenign FPR: {fpr:.4%}")

    # confusion matrix (row-normalized, same format as your 2017 runs)
    cm = confusion_matrix(y18, y_pred, labels=labels_in_use, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", values_format=".2f")
    ax.set_title(f"RF ({tag}) — CROSS-DATASET (2018)\nRow-normalized — diagonal = per-class recall")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(PROJECT/"results"/"figures"/f"cross_{tag}_confusion_matrix.png", dpi=150)
    plt.show()

    results[tag] = {"acc": acc, "macro_f1": mf1, "bacc": bacc, "mcc": mcc, "fpr": fpr}
    # keep per-class F1s for the drop table
    results[tag]["per_class_f1"] = {
        code_to_name[c]: f1_score(y18 == c, y_pred == c, average="binary")
        for c in labels_in_use}

# %% Stage 2 — THE DROP TABLE (Phase 3's deliverable)
# within-dataset F1 comes from your saved 2017 reports; cross F1 from this run.
def parse_report_f1(path):
    """Read per-class F1 from a stored classification report."""
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.strip().rsplit(None, 4)
        if len(parts) == 5 and parts[0] not in ("macro", "weighted", "accuracy") \
           and not parts[0].endswith("avg"):
            try:
                out[parts[0]] = float(parts[3])
            except ValueError:
                pass
    return out

WITHIN = {
    "rf_smote":      TABS/"rf_classification_report.txt",
    "rf_imbalanced": TABS/"imb_classification_report.txt",
}

rows = []
for tag in RF_MODELS:
    if tag not in results: continue
    within_f1 = parse_report_f1(WITHIN[tag])
    for cls in names:
        w = within_f1.get(cls)
        c = results[tag]["per_class_f1"].get(cls)
        if w is None or c is None: continue
        rows.append({"model": tag, "class": cls,
                     "within_f1": w, "cross_f1": round(c, 3),
                     "drop": round(w - c, 3)})

drop = pd.DataFrame(rows)
# rank by drop severity — worst transfer at the top (SHAP phase targets)
drop = drop.sort_values(["model", "drop"], ascending=[True, False])
drop.to_csv(TABS/"rf_cross_dataset_drop_table.csv", index=False)
print("\n" + "="*62)
print("PER-CLASS DROP TABLE (within-2017 F1 -> cross-2018 F1)")
print("="*62)
print(drop.to_string(index=False))

# %% Stage 3 — Summary: the one-screen verdict
print("\n" + "="*62)
print("REGIME COMPARISON — CROSS-DATASET")
print("="*62)
for tag, r in results.items():
    print(f"{tag:15s} Macro F1 {r['macro_f1']:.4f} | Acc {r['acc']:.4f} | "
          f"Benign FPR {r['fpr']:.4%}")
print("\nWorst-transfer classes (largest F1 drop), per model:")
for tag in results:
    worst = drop[drop["model"] == tag].head(5)
    print(f"\n{tag}:")
    print(worst[["class", "within_f1", "cross_f1", "drop"]].to_string(index=False))
# %%
