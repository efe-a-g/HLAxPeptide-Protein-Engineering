#!/usr/bin/env python3
"""
Aggregate ablation results: mean +/- spread per (config, variant, split), and
paired per-seed deltas against the BLOSUM+BLOSUM baseline *of the same variant*.

Two things this file is careful about:

1. VARIANTS MUST NOT BE POOLED. The same config appears at 25 epochs / 60 units
   (the base protocol) and again under phase-2 probes at 100 epochs, 256 units,
   or PCA-180. Grouping on (config, split) alone would silently average a
   100-epoch run into the 25-epoch number and corrupt every comparison. Each
   run therefore carries a `variant` key, and a baseline is only ever compared
   with an arm from the same variant.

2. DELTAS ARE PAIRED PER SEED, so both arms see the same split, and the
   per-seed differences are then averaged. On leave-allele-out the seed spread
   is ~61x the random-split spread (it is dominated by *which* 15 alleles are
   held out), so an unpaired comparison of means there would be pure noise.
"""
import argparse
import json

import numpy as np
import pandas as pd

METRICS = ["mean_allele_pcc", "mean_allele_scc", "global_pcc_score",
           "global_scc_score", "auc_1h", "auc_2h", "rmse_hours", "mae_hours"]
SPLIT_ORDER = ["supertype", "random", "cluster", "allele"]
BASE_VARIANT = "base"

PRETTY = {
    "blosum_pep": "BLOSUM-pep", "esm2_150m_pep_res": "ESM2-pep",
    "esm2_35m_pep_res": "ESM2(35M)-pep", "esm2_150m_pep_mean": "ESM2-pep(mean)",
    "none": "none",
    "blosum_hla": "BLOSUM-HLA", "boltz_BF": "Boltz-HLA(B+F)",
    "boltz_BFs": "Boltz-HLA(B+F+s)", "boltz_s": "Boltz-HLA(s)",
    "esm2_150m_hla_mean": "ESM2-HLA", "onehot_hla": "onehot-HLA",
    "peptide_mean_null": "peptide-only null",
}


def pretty_side(spec):
    """Render a possibly '+'-joined feature spec."""
    return " + ".join(PRETTY.get(p, p) for p in spec.split("+"))


def pretty(pep, hla):
    return f"{pretty_side(pep)} | {pretty_side(hla)}"


def variant_of(r):
    """Label the training protocol so distinct protocols are never pooled."""
    tag = r.get("tag") or ""
    if tag == "noise_init":
        return "noise_init"
    bits = []
    if r.get("epochs", 25) != 25:
        bits.append(f"ep{int(r['epochs'])}")
    if r.get("hidden_dim", 60) != 60:
        bits.append(f"h{int(r['hidden_dim'])}")
    if r.get("pca_dim", 0):
        bits.append(f"pca{int(r['pca_dim'])}")
    return "+".join(bits) if bits else BASE_VARIANT


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    for col, default in [("tag", ""), ("pca_dim", 0), ("epochs", 25),
                         ("hidden_dim", 60)]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["config"] = [pretty(p, h) for p, h in zip(df["pep"], df["hla"])]
    df["variant"] = [variant_of(r) for _, r in df.iterrows()]
    df = df.drop_duplicates(["config", "variant", "split", "seed"], keep="last")
    return df


def summarize(df):
    recs = []
    for (cfg, variant, split), g in df.groupby(["config", "variant", "split"]):
        rec = {"config": cfg, "variant": variant, "split": split,
               "n_features": int(g["n_features"].iloc[0]),
               "n_seeds": int(len(g)),
               "runtime_s": float(g["runtime_s"].mean())}
        for m in METRICS:
            rec[m] = float(g[m].mean())
            rec[m + "_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        recs.append(rec)
    return pd.DataFrame(recs)


def paired_deltas(df, baseline_cfg):
    """Paired per-seed deltas vs the baseline WITHIN each (variant, split)."""
    out = []
    for (variant, split), sub in df.groupby(["variant", "split"]):
        base = sub[sub["config"] == baseline_cfg].set_index("seed")
        if base.empty:
            continue
        for cfg, g in sub.groupby("config"):
            if cfg == baseline_cfg:
                continue
            g = g.set_index("seed")
            shared = sorted(set(g.index) & set(base.index))
            if not shared:
                continue
            rec = {"config": cfg, "variant": variant, "split": split,
                   "n_paired": len(shared)}
            for m in METRICS:
                d = g.loc[shared, m].to_numpy() - base.loc[shared, m].to_numpy()
                d = d[np.isfinite(d)]
                if len(d) == 0:
                    rec[f"d_{m}"] = np.nan
                    rec[f"d_{m}_sd"] = np.nan
                    rec[f"d_{m}_t"] = np.nan
                    continue
                sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
                se = sd / np.sqrt(len(d)) if len(d) > 1 else 0.0
                rec[f"d_{m}"] = float(np.mean(d))
                rec[f"d_{m}_sd"] = sd
                rec[f"d_{m}_t"] = float(np.mean(d) / se) if se > 1e-12 else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def print_block(summ, dd, variant, baseline_cfg, metric):
    s_all = summ[summ["variant"] == variant]
    if s_all.empty:
        return
    print(f"\n{'#'*108}\n# VARIANT: {variant}"
          f"{'   (25 epochs, 60 hidden units -- the published protocol)' if variant == BASE_VARIANT else ''}"
          f"\n{'#'*108}")
    for split in SPLIT_ORDER:
        s = s_all[s_all["split"] == split]
        if s.empty:
            continue
        asc = metric in ("rmse_hours", "mae_hours")
        s = s.sort_values(metric, ascending=asc)
        print(f"\n  SPLIT: {split}")
        print(f"  {'configuration':46s} {'dims':>5s} {'allele_pcc':>17s} "
              f"{'global_pcc':>17s} {'auc_1h':>16s} {'rmse_h':>13s}")
        for _, r in s.iterrows():
            print(f"  {r['config']:46s} {int(r['n_features']):5d} "
                  f"{r['mean_allele_pcc']:.4f}+-{r['mean_allele_pcc_sd']:.4f}  "
                  f"{r['global_pcc_score']:.4f}+-{r['global_pcc_score_sd']:.4f}  "
                  f"{r['auc_1h']:.4f}+-{r['auc_1h_sd']:.4f}  "
                  f"{r['rmse_hours']:6.2f}+-{r['rmse_hours_sd']:.2f}")
        if dd.empty:
            continue
        d = dd[(dd["variant"] == variant) & (dd["split"] == split)]
        if d.empty:
            continue
        d = d.sort_values("d_mean_allele_pcc", ascending=False)
        print(f"\n    paired delta vs '{baseline_cfg}' (|t|>2.5 marked *)")
        print(f"    {'configuration':46s} {'d_allele_pcc':>22s} {'d_global_pcc':>22s}")
        for _, r in d.iterrows():
            def cell(m):
                v, sd, t = r[f"d_{m}"], r[f"d_{m}_sd"], r[f"d_{m}_t"]
                if not np.isfinite(v):
                    return f"{'n/a':>22s}"
                star = "*" if np.isfinite(t) and abs(t) > 2.5 else " "
                return f"{v:+.4f}+-{sd:.4f} t={t:+6.2f}{star}" if np.isfinite(t) \
                    else f"{v:+.4f}+-{sd:.4f} t=   n/a "
            print(f"    {r['config']:46s} {cell('mean_allele_pcc')} {cell('global_pcc_score')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    ap.add_argument("--baseline", default="BLOSUM-pep | BLOSUM-HLA")
    ap.add_argument("--metric", default="mean_allele_pcc",
                    help="sort key; mean_allele_pcc is primary because global "
                         "PCC is inflated by between-allele variance")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--out_prefix", default="outputs/ablation/summary")
    args = ap.parse_args()

    df = load(args.results)
    print(f"[*] {len(df)} runs | {df['config'].nunique()} configs | "
          f"variants {sorted(df['variant'].unique())} | "
          f"seeds {sorted(int(s) for s in df['seed'].unique())}")

    summ = summarize(df)
    dd = paired_deltas(df, args.baseline)
    summ.to_csv(f"{args.out_prefix}_mean_std.csv", index=False)
    if not dd.empty:
        dd.to_csv(f"{args.out_prefix}_deltas.csv", index=False)

    variants = args.variants or (
        [BASE_VARIANT] + sorted(v for v in summ["variant"].unique()
                                if v not in (BASE_VARIANT, "noise_init")))
    for v in variants:
        print_block(summ, dd, v, args.baseline, args.metric)

    print(f"\n[+] wrote {args.out_prefix}_mean_std.csv and {args.out_prefix}_deltas.csv")


if __name__ == "__main__":
    main()
