#!/usr/bin/env python3
"""
Is the improvement real, given that the mean is taken over only 16 held-out alleles?

The headline metric averages a per-allele Spearman over 16 alleles, so it is a mean of 16
numbers and inherits their spread. A paired bootstrap over alleles -- resampling which alleles
are in the test set, keeping both models' predictions for each -- gives a confidence interval
for the *difference*, which is the quantity in question. Pairing matters: the two models see
the same alleles, and allele difficulty varies far more than the gap between models.

Baseline predictions come from outputs/target_transform (MLP, raw target, 90 epochs, the
reproduction of baseline.py at 0.4156); the candidate from outputs/final_recipe. Both are the
5-seed ensembles that each protocol actually produces.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def per_allele_scc(alleles, y_true, y_pred):
    out = {}
    frame = pd.DataFrame({"a": alleles, "t": y_true, "p": y_pred})
    for a, g in frame.groupby("a"):
        if len(g) >= 3 and g["t"].std() > 1e-6 and g["p"].std() > 1e-6:
            out[a] = spearmanr(g["t"], g["p"]).statistic
    return pd.Series(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--baseline_npz", default="outputs/target_transform/test_predictions.npz")
    p.add_argument("--baseline_key", default="raw")
    p.add_argument("--candidates", nargs="+",
                   default=["outputs/final_recipe/bilinear_rank/preds_bilinear_rank.npy",
                            "outputs/final_recipe/pssm_rank/preds_pssm_rank.npy"])
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="outputs/final_recipe")
    args = p.parse_args()

    z = np.load(args.baseline_npz, allow_pickle=True)
    alleles, s_te = z["alleles"], z["s_te"]
    base = per_allele_scc(alleles, s_te, z[args.baseline_key].mean(0))

    rows, table = [], {"baseline (mlp:raw, 90ep)": base}
    for path in args.candidates:
        if not os.path.isfile(path):
            print(f"[!] missing, skipped: {path}")
            continue
        name = os.path.basename(path).replace("preds_", "").replace(".npy", "")
        cand = per_allele_scc(alleles, s_te, np.load(path).mean(0))
        common = base.index.intersection(cand.index)
        b, c = base[common].to_numpy(), cand[common].to_numpy()
        d = c - b

        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, len(common), size=(args.n_boot, len(common)))
        boot = d[idx].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res = {"candidate": name, "n_alleles": int(len(common)),
               "baseline_mean_scc": float(b.mean()), "candidate_mean_scc": float(c.mean()),
               "mean_delta": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
               "p_boot_delta_le_0": float((boot <= 0).mean()),
               "n_alleles_improved": int((d > 0).sum())}
        rows.append(res)
        table[name] = cand
        print(f"{name:24s} {res['baseline_mean_scc']:.4f} -> {res['candidate_mean_scc']:.4f} | "
              f"delta {res['mean_delta']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] | "
              f"better on {res['n_alleles_improved']}/{res['n_alleles']} alleles | "
              f"P(delta<=0) {res['p_boot_delta_le_0']:.4f}")

    per = pd.DataFrame(table).sort_values(list(table)[-1], ascending=False)
    print("\nPer-allele Spearman:")
    print(per.round(3).to_string())
    per.to_csv(os.path.join(args.output_dir, "per_allele_comparison.csv"))
    with open(os.path.join(args.output_dir, "bootstrap.json"), "w") as f:
        json.dump({"n_boot": args.n_boot, "comparisons": rows}, f, indent=2)


if __name__ == "__main__":
    main()
