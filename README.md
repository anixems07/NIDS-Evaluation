# ML-Based Network Intrusion Detection under Class Imbalance and Realistic Evaluation

Machine-learning NIDS benchmark on **CICIDS2017** with a leakage-free evaluation protocol, plus a **cross-dataset generalization** study on **CSE-CIC-IDS2018**.

## What this project studies

1. **Class imbalance, measured honestly** — Random Forest and XGBoost are trained under two regimes:
   - Natural imbalanced training data
   - SMOTE-balanced training data, applied strictly *after* the train/test split

   Both regimes are evaluated on an **untouched real-ratio test set** using imbalance-aware metrics such as Macro-F1, Balanced Accuracy, MCC, per-class Precision/Recall/F1, and benign false-positive rate.

2. **Zero-day behavior** — The rarest attacks (**Heartbleed** and **Infiltration**) are held out of training entirely as a simulated zero-day probe.

3. **Cross-dataset transfer** — Models trained on CICIDS2017 are evaluated on **CSE-CIC-IDS2018** using a feature-aligned schema with approximately 76 shared CICFlowMeter statistics. A per-class performance-drop table is produced, and SHAP feature-attribution comparisons across training regimes explain *what changed*.

## Results so far — Random Forest

**Training:** SMOTE-balanced  
**Evaluation:** Untouched real-ratio test set  
**Test samples:** 446,624

| Metric | Value |
|---|---:|
| Accuracy | 0.9983 |
| Macro F1 | 0.9495 |
| MCC | 0.9937 |
| Balanced Accuracy | 0.9858 |
| Benign FPR | 0.142% |

### XGBoost

Under the same regime:

- **Accuracy:** 0.9979
- **Macro F1:** 0.7882

Imbalanced-regime and cross-dataset results are available in `results/tables/`.

> **Note:** Reported results should be regenerated and verified against the current experiment outputs before publication.

## Dataset

### CICIDS2017

**MachineLearningCSV.zip**

Official source:

https://www.unb.ca/cic/datasets/ids-2017.html

### CSE-CIC-IDS2018

Source: AWS Open Data Registry / Canadian Institute for Cybersecurity (CIC).

The datasets are **not committed to this repository** because of their size. Place the raw CICIDS2017 CSV files in:

```text
data/raw/, 2018 CSVs in data/raw_2018/.
```
## Repository structure

```
.
├── src/
│   ├── preprocessing
│   ├── training
│   ├── alignment
│   └── verification scripts
│
├── data/
│   ├── raw/                  # CICIDS2017 CSVs (gitignored)
│   ├── raw_2018/             # CSE-CIC-IDS2018 CSVs (gitignored)
│   └── processed/            # Processed data and splits (gitignored)
│
├── results/
│   ├── figures/              # Confusion matrices, feature importance, etc.
│   ├── tables/               # Metric reports, verdict tables, label mappings
│   └── models/               # Trained models (gitignored)
│
├── requirements.txt
└── README.md
```

## Experimental design

### The project evaluates two major training regimes:

```
                    CICIDS2017
                         |
              +----------+----------+
              |                     |
        Non-SMOTE                 SMOTE
              |                     |
        +-----+-----+         +-----+-----+
        |           |         |           |
       2017        2018      2017        2018
     testing     testing    testing     testing
```


## Methodological guarantees

- No target leakage: IP/port/timestamp identifiers dropped before training
- SMOTE applied only to the training partition, post-split — never to test data
- Evaluation on untouched real-ratio test data (no synthetic rows anywhere in
  evaluation sets)
- All reported numbers traceable to files in `results/tables/`

## References

Sharafaldin et al. 2018 (CICIDS2017) · Engelen et al. 2021 (dataset errors) ·
Chawla et al. 2002 (SMOTE) · Sarhan/Layeghy/Portmann (NF-v2, future work)