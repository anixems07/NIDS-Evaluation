# %% Stage 0 — Load the preprocessed data
from pathlib import Path
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef)

# ---------------- paths setup ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"
MODELS_DIR = BASE_DIR / "results" / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data splits
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")          # 800,000 x 46 — the textbook
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["label"] # 800,000 answers
X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")           # 446,624 x 46 — the exam
y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv")["label"]  # the answer key

mapping_path = TABLES_DIR / "label_mapping.csv"
if not mapping_path.exists():
    mapping_path = PROCESSED_DIR / "label_mapping.csv"
mapping = pd.read_csv(mapping_path)
code_to_name = dict(zip(mapping["code"], mapping["label_name"]))

print("Train:", X_train.shape, "| Test:", X_test.shape)
# %%
# %% Stage 1 — Train the Random Forest
model = RandomForestClassifier(
    n_estimators=100,   # 100 trees will vote
    random_state=42,    # anyone re-running this gets IDENTICAL results (reproducibility)
    n_jobs=-1,          # use every CPU core (fan will spin up — that's correct)
)

t0 = time.time()
model.fit(X_train, y_train)      # <-- this line IS the training
print(f"Training finished in {(time.time()-t0)/60:.1f} minutes")
# %%
# %% Stage 2 — Predict on unseen data + compute the metrics
t0 = time.time()
y_pred = model.predict(X_test)     # the model sits the exam
print(f"446,624 predictions in {time.time()-t0:.1f} seconds")

labels_in_use = sorted(y_test.unique())
names = [code_to_name[c] for c in labels_in_use]

report = classification_report(y_test, y_pred, labels=labels_in_use,
                                target_names=names, digits=3)
print(report)

acc   = accuracy_score(y_test, y_pred)
mf1   = f1_score(y_test, y_pred, average="macro")
wf1   = f1_score(y_test, y_pred, average="weighted")
bacc  = balanced_accuracy_score(y_test, y_pred)
mcc   = matthews_corrcoef(y_test, y_pred)

# benign false-positive rate: fraction of NORMAL traffic wrongly flagged
benign_code = int(mapping.loc[mapping.label_name == "BENIGN", "code"].iloc[0])
benign_fpr  = (y_pred[(y_test == benign_code).values] != benign_code).mean()

print(f"Accuracy           : {acc:.4f}")
print(f"Macro F1           : {mf1:.4f}   <- paper headline number")
print(f"Weighted F1        : {wf1:.4f}")
print(f"Balanced Accuracy  : {bacc:.4f}")
print(f"MCC                : {mcc:.4f}")
print(f"Benign FPR (false alarms on normal traffic): {benign_fpr:.4%}")

with open(TABLES_DIR / "classification_report.txt", "w") as f:
    f.write(report)
with open(TABLES_DIR / "metrics_summary.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nMacro F1: {mf1:.4f}\nWeighted F1: {wf1:.4f}\n"
            f"Balanced Accuracy: {bacc:.4f}\nMCC: {mcc:.4f}\nBenign FPR: {benign_fpr:.4%}")
# %% Stage 3 — Confusion matrix
cm = pd.crosstab(pd.Series(y_test, name="Actual").map(code_to_name),
                 pd.Series(y_pred, name="Predicted").map(code_to_name))
cm.to_csv(TABLES_DIR / "confusion_matrix.csv")
cm.to_csv(TABLES_DIR / "rf_confusion_matrix.csv")
print(cm)
# %% Stage 4 — Which features mattered?
imp = pd.Series(model.feature_importances_, index=X_train.columns)
top = imp.nlargest(15)
top.sort_values().plot(kind="barh", figsize=(8, 6),
                       title="Top 15 features used by the Random Forest")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
print(top.sort_values(ascending=False))
# %% Stage 5 — Persist the model
joblib.dump(model, MODELS_DIR / "rf_model.joblib")
print(f"Saved {MODELS_DIR / 'rf_model.joblib'}")
# %% Stage 3B — Confusion matrix visual (reloads the saved model)
saved_model = joblib.load(MODELS_DIR / "rf_model.joblib")
y_pred_visual = saved_model.predict(X_test)

# Keep every class in a stable, meaningful order, including classes with no
# predictions, so the CSV and heatmap use the same axes.
labels_in_use = sorted(y_test.unique())
names = [code_to_name[c] for c in labels_in_use]

cm_raw = confusion_matrix(y_test, y_pred_visual, labels=labels_in_use)
pd.DataFrame(cm_raw, index=names, columns=names).to_csv(
    TABLES_DIR / "rf_confusion_matrix.csv")

cm_norm = confusion_matrix(y_test, y_pred_visual, labels=labels_in_use,
                           normalize="true")
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay(cm_norm, display_labels=names).plot(
    ax=ax, cmap="Blues", values_format=".2f")
ax.set_title("Random Forest (SMOTE-balanced training)\n"
             "Row-normalized - diagonal = per-class recall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rf_confusion_matrix.png", dpi=150)
plt.close(fig)
print("Saved results/tables/rf_confusion_matrix.csv (counts) + "
      "results/figures/rf_confusion_matrix.png (visual)")
# %% Stage 6 — Zero-day probe: where do never-seen attacks land?
zd_file = PROCESSED_DIR / "X_zero_day.csv"
y_zd_file = PROCESSED_DIR / "y_zero_day.csv"
if zd_file.exists():
    X_zd = pd.read_csv(zd_file)
    zero_day_model = joblib.load(MODELS_DIR / "rf_model.joblib")
    zd_pred_names = pd.Series(zero_day_model.predict(X_zd)).map(code_to_name)

    counts = zd_pred_names.value_counts()
    print("Zero-day attack predictions:")
    print(counts)

    if y_zd_file.exists():
        y_zd = pd.read_csv(y_zd_file)["label"]
        probe = pd.crosstab(y_zd.map(code_to_name), zd_pred_names)
        print(probe)

        with open(TABLES_DIR / "rf_zero_day_results.txt", "w") as f:
            f.write(
                "Zero-day probe - Random Forest (SMOTE regime)\n\n"
                "Destination of never-seen attacks:\n"
                + counts.to_string()
                + "\n\nRows = true (hidden) class, columns = model's guess:\n"
                + probe.to_string()
            )
        print(f"Saved {TABLES_DIR / 'rf_zero_day_results.txt'}")