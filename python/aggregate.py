#!/usr/bin/env python3
"""
Aggregate ablation results: mean +/- spread per (config, split), and paired
per-seed deltas against the BLOSUM+BLOSUM baseline.

Why paired deltas: seed controls both the split draw and weight init, so arms
at the same seed see the same split. Comparing them per-seed and then averaging
removes split-to-split variance, which is much larger than the effect sizes
we are chasing. An unpaired comparison of means would hide real differences
under split noise -- and invent differences that are only split noise.
"""
import argparse
import json

import numpy as np
import pandas as pd

METRICS = ["mean_allele_pcc", "global_pcc_score", "global_scc_score",
           "auc_1h", "auc_2h", "rmse_hours", "mae_hours"]
SPLIT_ORDER = ["random", "cluster", "allele"]

PRETTY = {
    "blosum_pep": "BLOSUM-pep", "esm2_150m_pep_res": "ESM2-pep",
    "esm2_35m_pep_res": "ESM2(35M)-pep", "esm2_150m_pep_mean": "ESM2-pep(mean)",
    "none": "none",
    "blosum_hla": "BLOSUM-HLA", "boltz_BF": "Boltz-HLA(B+F)",
    "boltz_BFs": "Boltz-HLA(B+F+s)", "boltz_s": "Boltz-HLA(s)",
    "esm2_150m_hla_mean": "ESM2-HLA", "onehot_hla": "onehot-HLA",
}


def pretty(pep, hla):
    return f"{PRETTY.get(pep, pep)} + {PRETTY.get(hla, hla)}"


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    df["config"] = [pretty(p, h) for p, h in zip(df["pep"], df["hla"])]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    ap.add_argument("--baseline", default="BLOSUM-pep + BLOSUM-HLA")
    ap.add_argument("--out_prefix", default="outputs/ablation/summary")
    args = ap.parse_args()

    df = load(args.results)
    df = df.drop_duplicates(["config", "split", "seed", "tag", "pca_dim"],
                            keep="last")
    print(f"[*] {len(df)} runs | {df['config'].nunique()} configs | "
          f"seeds {sorted(df['seed'].unique())}")

    # ---- mean +/- std table ------------------------------------------------
    g = df.groupby(["config", "split"])
    summ = g[METRICS].agg(["mean", "std", "count"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    summ["n_features"] = g["n_features"].first().values
    summ["split"] = pd.Categorical(summ["split"], SPLIT_ORDER, ordered=True)
    summ = summ.sort_values(["split", "global_pcc_score_mean"],
                            ascending=[True, False])
    summ.to_csv(f"{args.out_prefix}_mean_std.csv", index=False)

    # ---- paired per-seed deltas vs baseline --------------------------------
    deltas = []
    for split in df["split"].unique():
        sub = df[df["split"] == split]
        base = sub[sub["config"] == args.baseline].set_index("seed")
        if base.empty:
            continue
        for cfg, grp in sub.groupby("config"):
            if cfg == args.baseline:
                continue
            grp = grp.set_index("seed")
            shared = sorted(set(grp.index) & set(base.index))
            if not shared:
                continue
            rec = {"config": cfg, "split": split, "n_paired": len(shared)}
            for m in METRICS:
                d = grp.loc[shared, m].to_numpy() - base.loc[shared, m].to_numpy()
                rec[f"d_{m}_mean"] = float(np.mean(d))
                rec[f"d_{m}_std"] = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
                # t-like effect size: mean delta / standard error of the mean
                se = (np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
                rec[f"d_{m}_t"] = float(np.mean(d) / se) if se and se > 1e-12 else np.nan
            deltas.append(rec)
    dd = pd.DataFrame(deltas)
    if not dd.empty:
        dd["split"] = pd.Categorical(dd["split"], SPLIT_ORDER, ordered=True)
        dd = dd.sort_values(["split", "d_global_pcc_score_mean"],
                            ascending=[True, False])
        dd.to_csv(f"{args.out_prefix}_deltas.csv", index=False)

    # ---- console report ----------------------------------------------------
    for split in SPLIT_ORDER:
        s = summ[summ["split"] == split]
        if s.empty:
            continue
        print(f"\n{'='*104}\nSPLIT: {split}\n{'='*104}")
        print(f"{'config':40s} {'dims':>6s} {'allele_pcc':>18s} "
              f"{'global_pcc':>18s} {'auc_1h':>16s} {'rmse_h':>14s}")
        for _, r in s.iterrows():
            print(f"{r['config']:40s} {int(r['n_features']):6d} "
                  f"{r['mean_allele_pcc_mean']:.4f}+-{r['mean_allele_pcc_std']:.4f}  "
                  f"{r['global_pcc_score_mean']:.4f}+-{r['global_pcc_score_std']:.4f}  "
                  f"{r['auc_1h_mean']:.4f}+-{r['auc_1h_std']:.4f}  "
                  f"{r['rmse_hours_mean']:6.2f}+-{r['rmse_hours_std']:.2f}")

        if dd.empty:
            continue
        d = dd[dd["split"] == split]
        if d.empty:
            continue
        print(f"\n  paired per-seed delta vs '{args.baseline}' "
              f"(delta global_pcc; |t|>2.5 marked *):")
        for _, r in d.iterrows():
            t = r["d_global_pcc_score_t"]
            star = " *" if np.isfinite(t) and abs(t) > 2.5 else "  "
            print(f"    {r['config']:40s} n={int(r['n_paired'])} "
                  f"{r['d_global_pcc_score_mean']:+.4f} "
                  f"+-{r['d_global_pcc_score_std']:.4f} "
                  f"(t={t:+.2f}){star}   "
                  f"d_allele_pcc {r['d_mean_allele_pcc_mean']:+.4f}")

    print(f"\n[+] wrote {args.out_prefix}_mean_std.csv "
          f"and {args.out_prefix}_deltas.csv")


if __name__ == "__main__":
    main()
