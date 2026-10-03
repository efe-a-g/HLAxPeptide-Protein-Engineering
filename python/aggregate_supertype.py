#!/usr/bin/env python3
"""
Summarise outputs/supertype/results.jsonl -- the runs that comply with the
project-wide convention in baseline.py (supertype-stratified leave-allele-out,
mean per-allele Spearman).

Deltas are paired per seed against the BLOSUM baseline, so both arms see the
same held-out alleles. On a leave-allele-out split that pairing is essential:
which alleles are drawn moves the score far more than the features do.
"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

BASE = "blosum_pep | blosum_hla"
PRETTY = {
    "blosum_pep | blosum_hla": "BLOSUM-pep | BLOSUM-HLA  (baseline)",
    "blosum_pep | boltz_BF": "BLOSUM-pep | Boltz-HLA  (replace)",
    "blosum_pep | blosum_hla+boltz_BF": "BLOSUM-pep | BLOSUM+Boltz  (augment)",
    "blosum_pep | esm2_150m_hla_mean": "BLOSUM-pep | ESM2-HLA  (replace)",
    "blosum_pep | onehot_hla": "BLOSUM-pep | onehot-HLA  (control)",
    "esm2_150m_pep_res | blosum_hla": "ESM2-pep | BLOSUM-HLA",
    "esm2_150m_pep_res | boltz_BF": "ESM2-pep | Boltz-HLA",
    "peptide-only null": "peptide-only null  (no allele info)",
}
M = ["mean_allele_scc", "mean_allele_pcc", "median_allele_scc", "global_scc"]

rows = [json.loads(l) for l in open("outputs/supertype/results.jsonl") if l.strip()]
df = pd.DataFrame(rows).drop_duplicates(["label", "seed"], keep="last")
print(f"[*] {len(df)} runs | {df['label'].nunique()} arms | "
      f"seeds {sorted(df['seed'].unique())} | {int(df['epochs'].max())} epochs")
print(f"[*] split: supertype-stratified leave-allele-out, "
      f"{int(df['n_train'].iloc[0]):,} train / {int(df['n_test'].iloc[0]):,} test, "
      f"{int(df['n_alleles_scored'].iloc[0])} alleles scored\n")

g = df.groupby("label")
summ = g[M].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ["n"] = g.size()
summ["dims"] = g["n_features"].first()
summ = summ.sort_values("mean_allele_scc_mean", ascending=False)

print(f"{'arm':40s} {'dims':>5s} {'mean per-allele SCC':>22s} "
      f"{'mean per-allele PCC':>22s} {'median SCC':>16s}")
print("-" * 112)
base_row = summ.loc[BASE] if BASE in summ.index else None
for label, r in summ.iterrows():
    name = PRETTY.get(label, label)
    mark = " <<" if label == BASE else ""
    print(f"{name:40s} {int(r['dims']):5d} "
          f"{r['mean_allele_scc_mean']:.4f}+-{r['mean_allele_scc_std']:.4f}       "
          f"{r['mean_allele_pcc_mean']:.4f}+-{r['mean_allele_pcc_std']:.4f}       "
          f"{r['median_allele_scc_mean']:.4f}{mark}")

print(f"\n\nPaired per-seed delta vs baseline (mean per-allele SCC), * = |t| > 2.5")
print("-" * 112)
b = df[df.label == BASE].set_index("seed")
out = []
for label, grp in df.groupby("label"):
    if label == BASE:
        continue
    a = grp.set_index("seed")
    sh = sorted(set(a.index) & set(b.index))
    d = a.loc[sh, "mean_allele_scc"].to_numpy() - b.loc[sh, "mean_allele_scc"].to_numpy()
    d = d[np.isfinite(d)]
    if len(d) < 2:
        continue
    sd = d.std(ddof=1); se = sd / np.sqrt(len(d))
    t = d.mean() / se if se > 1e-12 else np.nan
    out.append((d.mean(), sd, t, len(d), label))
for mu, sd, t, n, label in sorted(out, reverse=True):
    star = "*" if abs(t) > 2.5 else " "
    print(f"{PRETTY.get(label, label):40s} {mu:+.4f} +-{sd:.4f}  "
          f"t={t:+6.2f}{star}  n={n}")

# How much does the baseline beat the null that knows nothing about alleles?
nl = df[df.label == "peptide-only null"].set_index("seed")
sh = sorted(set(b.index) & set(nl.index))
d = b.loc[sh, "mean_allele_scc"].to_numpy() - nl.loc[sh, "mean_allele_scc"].to_numpy()
sd = d.std(ddof=1); t = d.mean() / (sd / np.sqrt(len(d)))
print(f"\nbaseline vs peptide-only null: {d.mean():+.4f} +-{sd:.4f} "
      f"t={t:+.2f}{'*' if abs(t) > 2.5 else ''}  "
      f"-- the margin that allele modelling actually buys")
