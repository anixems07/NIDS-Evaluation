# Machine Learning-Based Network Intrusion Detection under Class Imbalance, Zero-Day Probing, and Cross-Dataset Domain Shift

A leak-free, reproducible machine learning benchmark on **CICIDS2017** with a **cross-dataset generalization and explainability study** on **CSE-CIC-IDS2018**.

---

## Abstract and Research Motivation

Machine Learning-based Network Intrusion Detection Systems (ML-NIDS) frequently report near-perfect detection metrics (>99.8% accuracy) in contemporary literature. However, subsequent independent analyses have demonstrated that these figures are frequently driven by methodological artifacts rather than genuine detection capabilities:

1. **Identifier and Temporal Leakage:** Retaining socket-level identifiers (Source IP, Destination IP, Source Port, Destination Port, Timestamp, Flow ID) allows classifiers to memorize network topology and testbed-specific artifacts rather than learning protocol-invariant behavioral dynamics.
2. **Evaluation on Synthetic Distributions:** Applying oversampling techniques (such as SMOTE) prior to dataset splitting or evaluating models on synthetically balanced test sets distorts operating characteristics, severely underestimating operational false alarm rates.
3. **Closed-World Evaluation Limits:** Standard evaluations operate under a closed-world assumption where all test attack classes are observed during training, leaving the vulnerability to novel zero-day threats unquantified.
4. **Cross-Dataset Brittleness:** High in-distribution performance on a single testbed capture does not guarantee generalizability across distinct network environments, architectures, and traffic profiles.

This project implements a leak-free evaluation protocol addressing these challenges through four primary investigations:
- **Class Imbalance Dynamics:** A controlled comparison between natural imbalanced training and SMOTE-balanced training evaluated on an **untouched real-ratio test set** (446,624 flows).
- **Simulated Zero-Day Attack Probe:** Complete sequestration of rare exploit families (`Heartbleed` and `Infiltration`) during training to evaluate open-world detection limits.
- **Cross-Dataset Generalization:** Transfer evaluation of models trained on CICIDS2017 to **CSE-CIC-IDS2018** (2,931,071 flows) via an aligned 46-feature behavioral schema.
- **Explainable AI (XAI) Attribution Diagnostics:** Utilizing **SHAP (SHapley Additive exPlanations)** to analyze whether cross-dataset performance drops stem from feature attribution shifts or domain-level distribution drift.

---

## Experimental Architecture

```
                          CICIDS2017 Raw Data (8 CSVs, 2,830,743 flows)
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      ▼                                                   ▼
            Data Hygiene Pipeline                                Zero-Day Sequestration
      • Socket ID removal (IP, Port, Time)                       • Heartbleed (11 flows)
      • Inf/NaN removal (2,867 flows)                            • Infiltration (36 flows)
      • Deduplication (594,712 flows)                                     │
      • Zero-variance pruning (8 features)                                ▼
      • Pearson correlation pruning (ρ > 0.95, 23 features)      Zero-Day Probe (47 flows)
                      │
                      ▼
        Sanitized Dataset: 2,233,164 flows × 46 features
                      │
        Stratified 80/20 Train/Test Partitioning
                      │
        ┌─────────────┴───────────────────────────────────────────────────┐
        ▼                                                                 ▼
Training Partition (80%: 1,786,493 flows)               Untouched Test Partition (20%: 446,624 flows)
        │                                                                 │
   ┌────┴───────────────────────────┐                                     │
   ▼                                ▼                                     │
Regime A: Natural Imbalance    Regime B: SMOTE-Balanced                   │
• Native class distribution    • Majority downsampled to 300,000          │
• No synthetic oversampling    • Minority classes oversampled to 50,000   │
   │                                │                                     │
   ▼                                ▼                                     │
Random Forest Classifier       Random Forest Classifier                   │
   │                                │                                     │
   └───────────────┬────────────────┘                                     │
                   ▼                                                      │
   In-Distribution Evaluation (CICIDS2017) ◄──────────────────────────────┘
                   │
                   ├──────────────────────────────────────────────────────┐
                   ▼                                                      ▼
   Zero-Day Probe Evaluation                             Cross-Dataset Evaluation (CSE-CIC-IDS2018)
   • 47 unseen exploit flows                             • 2,931,071 flows aligned to 46-feature schema
                   │                                                      │
                   └──────────────────────────┬───────────────────────────┘
                                              ▼
                                 SHAP XAI Attribution Diagnosis
                                 • Spearman rank correlation (ρ)
                                 • Top-10 Jaccard feature overlap
                                 • Attribution shift vs. distribution drift
```

---

## Empirical Benchmark Results

All quantitative outcomes reported below are traceable to reproducible artifacts preserved in [`results/tables/`](results/tables/) and [`results/figures/`](results/figures/).

### 1. Training Regime Comparison: SMOTE vs. Natural Imbalance (CICIDS2017)

Evaluated on an untouched holdout partition of **446,624 flows** containing native class distributions (84.93% Benign, 15.07% Attacks across 11 known classes):

| Metric | SMOTE-Balanced | Natural Imbalance | Absolute Difference ($\Delta$) | Operational Significance |
|:---|:---:|:---:|:---:|:---|
| **Accuracy** | 0.9983 | **0.9989** | +0.0006 | Natural imbalance yields higher overall accuracy |
| **Macro F1-Score** | 0.9495 | **0.9595** | +0.0100 | Natural imbalance minimizes false alarms on prevalent classes |
| **Weighted F1-Score** | 0.9984 | **0.9989** | +0.0005 | Both regimes achieve high population-level accuracy |
| **Balanced Accuracy** | **0.9858** | 0.9492 | -0.0366 | **SMOTE yields superior recall across severe minority attacks** |
| **Matthews Corr. Coeff. (MCC)** | 0.9937 | **0.9960** | +0.0023 | High correlation between true and predicted classes |
| **Benign False Positive Rate (FPR)** | 0.1421% | **0.0467%** | -0.0954% | **Natural imbalance reduces operational false alarms by ~3x** |

#### Key Takeaways:
- **Minority Class Detection:** In the SMOTE regime, rare attack recall is substantially higher (e.g., Bot recall reaches 92.7%), but precision is penalized (50.6%) due to benign background confusion.
- **Operational Feasibility:** In high-throughput Security Operations Centers (SOCs), the natural imbalanced model produces an FPR of 0.0467% (versus 0.1421% for SMOTE), significantly mitigating operator alert fatigue.

---

### 2. Zero-Day Vulnerability Assessment

To evaluate whether supervised classifiers can detect unobserved attack families, **47 flows** representing `Heartbleed` (OpenSSL TLS vulnerability exploit) and `Infiltration` (multi-stage internal compromise) were isolated prior to preprocessing and evaluated against the trained classifiers:

| Target Attack Class | Evaluated Flows | Predicted as BENIGN | Predicted as Attack | Detection Rate |
|:---|:---:|:---:|:---:|:---:|
| **Heartbleed** | 11 | 11 (100.0%) | 0 (0.0%) | **0.00%** |
| **Infiltration** | 36 | 36 (100.0%) | 0 (0.0%) | **0.00%** |
| **Combined Zero-Day Probe** | **47** | **47 (100.0%)** | **0 (0.0%)** | **0.00%** |

#### Key Takeaways:
- Closed-world supervised models partition feature space solely to discriminate between known signatures.
- Unseen attack patterns exhibiting subtle protocol-compliant footprints default entirely to the majority `BENIGN` class distribution, demonstrating that supervised learning alone provides zero defense against novel exploits.

---

### 3. Cross-Dataset Generalization (CICIDS2017 ➔ CSE-CIC-IDS2018)

Models trained on CICIDS2017 were evaluated on **2,931,071 flows** from CSE-CIC-IDS2018 harmonized to the identical 46-feature schema:

| Attack Class | Within-2017 $F_1$ | Cross-2018 $F_1$ (Imbalanced) | Cross-2018 $F_1$ (SMOTE) | Performance Delta ($\Delta F_1$) | Generalization Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **DDoS** | 0.999 | 0.000 | 0.000 | -0.999 | Complete Failure |
| **DoS Hulk** | 0.998 | 0.000 | 0.000 | -0.998 | Complete Failure |
| **FTP-Patator** | 0.996 | 0.000 | 0.000 | -0.996 | Complete Failure |
| **SSH-Patator** | 0.995 | 0.000 | 0.000 | -0.995 | Complete Failure |
| **DoS Slowhttptest** | 0.967 | 0.000 | 0.000 | -0.967 | Complete Failure |
| **BENIGN** | 0.999 | 0.211 | 0.236 | -0.763 | Severe Degradation |
| **DoS GoldenEye** | 0.990 | 0.289 | 0.483 | -0.507 | Partial Retention |
| **DoS slowloris** | 0.995 | **0.865** | **0.847** | **-0.148** | **Robust Generalization** |

#### Root Cause of Generalization Collapse:
Statistical validation in `results/tables/alignment_verification.txt` highlights extreme inter-dataset distribution shifts across identical feature definitions:
- `Fwd IAT Min`: 2017 median = **3.0 $\mu s$** vs. 2018 median = **446.0 $\mu s$** (**148.7x shift**)
- `Flow IAT Min`: 2017 median = **4.0 $\mu s$** vs. 2018 median = **408.0 $\mu s$** (**102.0x shift**)
- `Init_Win_bytes_forward`: 2017 median = **235 bytes** vs. 2018 median = **8192 bytes** (**34.9x shift**)
- `Flow Duration`: 2017 median = **76,860 $\mu s$** vs. 2018 median = **11,201 $\mu s$** (**0.15x shift**)

---

### 4. SHAP Feature Attribution Diagnosis

Feature attribution analysis was conducted using `shap.TreeExplainer` across four configurations: `{SMOTE, Imbalanced} × {Within-2017, Cross-2018}`.

- **Volumetric Attacks (DDoS):** Feature ranking agreement between within-dataset and cross-dataset contexts remained high (Spearman $\rho = 0.86$–$0.93$; Top-10 Jaccard overlap = 80–90%). The classifier identifies the same behavioral features (`Flow IAT Min`, `Flow Bytes/s`), but cross-testbed distribution drift shifts feature values outside learned decision boundaries.
- **Brute Force Attacks (FTP-Patator):** Feature ranking agreement degraded to $\rho = 0.75$, with Top-10 overlap dropping to 50–60%. OS-level networking artifacts (such as `Init_Win_bytes_backward`) lost discriminatory power across environments.
- **Application-Layer Slow Attacks (DoS slowloris):** Maintained $F_1 \approx 0.85$ across datasets because low-and-slow HTTP connection exhaustion produces flow-timing characteristics that remain invariant to host stack differences.

Full diagnostic reports and multi-panel figures are available in [`results/tables/shap_findings_report.md`](results/tables/shap_findings_report.md) and [`results/figures/`](results/figures/).

---

## Methodological Rigor and Data Hygiene

1. **Target and Identifier Leakage Elimination:** Stripped `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Timestamp`, and `Flow ID`.
2. **Deterministic Data Cleaning:**
   - Removed 2,867 non-finite (`NaN`/`inf`) records.
   - Removed 594,712 duplicate flows to prevent cross-partition memorization.
3. **Dimensionality Reduction:**
   - Pruned 8 static zero-variance features (`Bwd PSH Flags`, `Fwd Avg Bytes/Bulk`, etc.).
   - Pruned 23 highly collinear features ($r > 0.95$) to prevent multicollinearity distortion.
   - Final sanitized feature set: 46 behavioral attributes.
4. **Leakage-Free Partitioning:**
   - Stratified 80/20 train/test splitting was executed **prior to any balancing transformations**.
   - Balancing (SMOTE / Undersampling) was applied **strictly to the 80% training partition**.
   - Test evaluation was conducted exclusively on untouched real-ratio data (0% synthetic rows).

---

## Repository Structure

```text
.
├── src/
│   ├── preprocess_v2.py          # Data ingestion, leakage elimination, correlation pruning, splits
│   ├── train_rf.py               # Supervised Random Forest training (SMOTE regime)
│   ├── train_rf_imbalanced.py    # Supervised Random Forest training (Natural imbalanced regime)
│   ├── align_2018.py             # Schema alignment and feature mapping for CSE-CIC-IDS2018
│   ├── verify_clean.py           # Integrity and split validation for processed CICIDS2017 data
│   ├── verify_alignment.py       # Structural and distribution verification for CSE-CIC-IDS2018
│   ├── evaluate_cross_rf.py      # Cross-dataset evaluation and performance drop calculations
│   └── shap_diagnosis.py         # SHAP TreeExplainer feature attribution and drift diagnostics
│
├── results/
│   ├── figures/                  # Confusion matrices, feature importances, and SHAP diagnostic panels
│   ├── tables/                   # Metric summaries, classification reports, CSV stats, and SHAP reports
│   └── models/                   # Model persistence directory (.gitkeep; binaries excluded)
│
├── requirements.txt              # Pinned Python package dependencies
├── .gitignore                    # Git exclusion rules for large datasets, binaries, logs, and environments
└── README.md                     # Research documentation and benchmark summary
```

---

## Installation and Reproduction

### 1. Environment Setup

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/anixems07/NIDS-Evaluation.git
cd NIDS-Evaluation

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Acquisition

The datasets used in this study are provided by the Canadian Institute for Cybersecurity:
- **CICIDS2017:** Download `MachineLearningCSV.zip` from [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and place the 8 CSV files in:
  ```text
  data/raw/
  ```
- **CSE-CIC-IDS2018 (Optional for Cross-Dataset):** Download the CSV files from the [AWS Open Data Registry](https://registry.opendata.aws/cse-cic-ids2018/) and place them in:
  ```text
  data/raw_2018/
  ```

### 3. Pipeline Execution

#### Stage 1: Data Preprocessing and Leakage Elimination
```bash
python src/preprocess_v2.py
python src/verify_clean.py
```

#### Stage 2: Supervised Baseline Modeling
```bash
# Evaluate SMOTE-balanced training regime
python src/train_rf.py

# Evaluate natural imbalanced training regime
python src/train_rf_imbalanced.py
```

#### Stage 3: Cross-Dataset Alignment and Transfer Evaluation
```bash
# Harmonize CSE-CIC-IDS2018 features with 2017 schema
python src/align_2018.py
python src/verify_alignment.py

# Evaluate cross-dataset transfer performance
python src/evaluate_cross_rf.py
```

#### Stage 4: Explainability and SHAP Attribution Diagnostics
```bash
python src/shap_diagnosis.py
```

---

## References

1. **CICIDS2017 Benchmark:**  
   I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, *"Toward Generating a New Dataset for Intrusion Detection and Realistic Evaluation,"* in Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018.
2. **Methodological Pitfalls in NIDS Datasets:**  
   G. Engelen, V. Rimmer, and W. Joosen, *"Troubleshooting an Intrusion Detection Dataset: The CICIDS2017 Case Study,"* in 2021 IEEE Security and Privacy Workshops (SPW), 2021.
3. **SMOTE Balancing:**  
   N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, *"SMOTE: Synthetic Minority Over-sampling Technique,"* Journal of Artificial Intelligence Research (JAIR), vol. 16, pp. 321–357, 2002.
4. **SHAP Feature Attribution:**  
   S. M. Lundberg and S.-I. Lee, *"A Unified Approach to Interpreting Model Predictions,"* in Advances in Neural Information Processing Systems (NeurIPS 30), 2017.