# ============================================================
# shap_diagnosis.py — Phase 4: WHY does transfer fail, per class?
# 4 contexts: {smote, imbalanced} x {within-2017, cross-2018}
# Outputs: per-context rankings (CSV), comparison table (CSV),
# 2x2 panel figures (PNG), plain-language findings report (MD).
# ============================================================

# %% Stage 0 — data, models, target classes
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
DATA, TABS, FIGS, MODELS = (PROJECT/"data"/"processed", PROJECT/"results"/"tables",
                            PROJECT/"results"/"figures", PROJECT/"results"/"models")

SAMPLE_PER_CLASS, TOP_K = 2000, 10

X17 = pd.read_csv(DATA/"X_test.csv");           y17 = pd.read_csv(DATA/"y_test.csv")["label"]
X18 = pd.read_csv(DATA/"X_test_2018_aligned.csv")
y18 = pd.read_csv(DATA/"y_test_2018_aligned.csv")["label"]
mapping = pd.read_csv(TABS/"label_mapping.csv")
name_to_code = dict(zip(mapping["label_name"], mapping["code"]))

MODELS_ = {"smote": MODELS/"rf_model.joblib",
           "imbalanced": MODELS/"rf_model_imbalanced.joblib"}
CONTEXTS = {"within2017": (X17, y17), "cross2018": (X18, y18)}

# targets: 3 worst-transfer classes from the drop table + 1 survivor as contrast
drop_path = TABS/"rf_cross_dataset_drop_table.csv"
if drop_path.exists():
    agg = pd.read_csv(drop_path).groupby("class")["drop"].max().sort_values(ascending=False)
    TARGETS = list(agg.head(3).index) + [agg.tail(1).index[0]]
else:
    TARGETS = ["DoS slowloris", "DoS GoldenEye", "Web Attack", "DDoS"]
print("SHAP target classes:", TARGETS, flush=True)

# %% Stage 1 — mean|SHAP| ranking per (class, model, context)
def class_shap(explainer, X_s, class_idx):
    sv = explainer.shap_values(X_s)
    if isinstance(sv, list):            return np.asarray(sv[class_idx])   # older shap
    sv = np.asarray(sv)
    if sv.ndim == 3:                    return sv[:, :, class_idx]         # newer shap
    return sv

rankings = {}
for tag, mpath in MODELS_.items():
    model = joblib.load(mpath)
    explainer = shap.TreeExplainer(model)
    order = list(model.classes_)
    for cls in TARGETS:
        code = name_to_code[cls]
        if code not in order: continue
        cidx = order.index(code)
        for cname, (Xc, yc) in CONTEXTS.items():
            rows = Xc[yc.values == code]
            if len(rows) == 0:
                print(f"[skip] {cls} absent in {cname}", flush=True); continue
            X_s = rows.sample(n=min(SAMPLE_PER_CLASS, len(rows)), random_state=42)
            t0 = time.time()
            sv = class_shap(explainer, X_s, cidx)
            imp = pd.Series(np.abs(sv).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
            rankings[(cls, tag, cname)] = imp
            imp.round(6).rename("mean_abs_shap").to_csv(
                TABS/f"shap_rank__{cls.replace(' ','_')}__{tag}__{cname}.csv")
            print(f"{cls:16s} {tag:10s} {cname:11s} top-5: {list(imp.head(5).index)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

# %% Stage 2 — comparison metrics (the paper's core table)
def jac(a, b, k=TOP_K):
    return len(set(a.head(k).index) & set(b.head(k).index)) / k
def rho_between(r1, r2):
    feats = list(r1.index)
    return spearmanr(r1.values, r2.loc[feats].values)[0]
def movers(r1, r2, n=5):
    p1 = pd.Series(range(1, len(r1)+1), index=r1.index)
    p2 = pd.Series(range(1, len(r2)+1), index=r2.index).reindex(r1.index)
    d = (p2 - p1).sort_values()
    return list(d.head(n).index), list(d.tail(n).index)   # gained, lost importance

COMPARISONS = [
    ("transfer_shift_SMOTE", ("smote","within2017"),      ("smote","cross2018")),
    ("transfer_shift_IMB",   ("imbalanced","within2017"), ("imbalanced","cross2018")),
    ("smote_effect_WITHIN",  ("smote","within2017"),      ("imbalanced","within2017")),
    ("smote_effect_CROSS",   ("smote","cross2018"),       ("imbalanced","cross2018")),
]
comp_metrics, rows = {}, []
for cls in TARGETS:
    for cname, (m1,c1), (m2,c2) in COMPARISONS:
        k1, k2 = (cls,m1,c1), (cls,m2,c2)
        if k1 not in rankings or k2 not in rankings: continue
        r1, r2 = rankings[k1], rankings[k2]
        rho, j = rho_between(r1, r2), jac(r1, r2)
        gained, lost = movers(r1, r2)
        comp_metrics[(cls, cname)] = {"rho": rho, "jac": j, "gained": gained, "lost": lost,
                                      "r1": r1, "r2": r2}
        rows.append({"class": cls, "comparison": cname, "spearman_rho": round(rho,3),
                     f"top{TOP_K}_jaccard": round(j,2),
                     f"top5_{c1}": ", ".join(map(str, r1.head(5).index)),
                     f"top5_{c2}": ", ".join(map(str, r2.head(5).index)),
                     "gained_importance": ", ".join(gained), "lost_importance": ", ".join(lost)})
comp = pd.DataFrame(rows)
comp.to_csv(TABS/"shap_comparison_table.csv", index=False)
print("\n" + comp.to_string(index=False), flush=True)

# %% Stage 3 — 2x2 panel figure per class (paper-ready)
import matplotlib.pyplot as plt
for cls in TARGETS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    slots = [("smote","within2017"), ("smote","cross2018"),
             ("imbalanced","within2017"), ("imbalanced","cross2018")]
    for ax, (tag, cn) in zip(axes.flat, slots):
        key = (cls, tag, cn)
        if key not in rankings: ax.axis("off"); continue
        imp = rankings[key].head(15)[::-1]
        ax.barh(imp.index, imp.values, color="steelblue")
        ax.set_title(f"{tag} — {cn}", fontsize=11)
    fig.suptitle(f"SHAP attribution — {cls}\n(bar length = average influence on the model's verdict)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS/f"shap_panel__{cls.replace(' ','_')}.png", dpi=150); plt.show()

# %% Stage 4 — auto-written plain-language findings (the writers' handoff)
def band(r):
    return ("very stable" if r >= 0.9 else "mostly stable" if r >= 0.7
            else "moderate shift" if r >= 0.5 else "MAJOR shift")
med17, med18 = X17.median(), X18.median()
F = ["# SHAP Diagnosis — Plain-Language Findings (auto-generated)",
     "Every number is traceable to a CSV in results/tables.\n",
     "## How to read the metrics",
     "- mean |SHAP| = average influence of a feature on the model's verdict for a class.",
     "  The ranking IS the model's 'evidence list' for that class.",
     "- Spearman rho = agreement between two evidence lists (1 = identical reasoning,",
     "  0 = completely different). Bands: >=0.9 very stable; 0.7-0.9 mostly stable;",
     "  0.5-0.7 moderate; <0.5 major shift.",
     f"- Top-{TOP_K} Jaccard = fraction of the most important features the two lists share.\n"]
if drop_path.exists():
    d = pd.read_csv(drop_path)
    for cls in TARGETS:
        F.append(f"\n## {cls}")
        for _, r in d[d["class"] == cls].iterrows():
            F.append(f"- Performance: {r['model']} — within-2017 F1 {r['within_f1']:.3f} "
                     f"-> cross-2018 F1 {r['cross_f1']:.3f} (drop {r['drop']:+.3f})")
        for cname, _, _ in COMPARISONS:
            m = comp_metrics.get((cls, cname))
            if not m: continue
            F.append(f"- {cname}: rho = {m['rho']:.2f} ({band(m['rho'])}), "
                     f"top-{TOP_K} overlap {m['jac']:.0%}")
            F.append(f"  evidence gained cross/context: {', '.join(m['gained'])}")
            F.append(f"  evidence lost: {', '.join(m['lost'])}")
        # supplementary: link attribution shift to measured distribution drift
        k1, k2 = (cls,"smote","within2017"), (cls,"smote","cross2018")
        if k1 in rankings and k2 in rankings:
            p1 = pd.Series(range(1,47), index=rankings[k1].index)
            p2 = pd.Series(range(1,47), index=rankings[k2].index).reindex(p1.index)
            shift = (p2 - p1).abs()
            valid = (med17 > 0) & (med18 > 0)
            drift = np.log2(med18[valid] / med17[valid]).abs().reindex(p1.index).fillna(0)
            r = spearmanr(shift[valid], drift[valid])[0]
            F.append(f"- Attribution-shift vs distribution-drift correlation: rho = {r:.2f} "
                     "(do the features whose importance moved most also drift most between networks?)")
(TABS/"shap_findings_report.md").write_text("\n".join(F))
print("\nSaved: shap_comparison_table.csv, shap_findings_report.md, "
      "shap_panel__<class>.png x", len(TARGETS), flush=True)