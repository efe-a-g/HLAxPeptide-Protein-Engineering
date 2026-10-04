#!/usr/bin/env python3
"""
Nearest-neighbour baseline: no training. For each held-out (allele, peptide) pair, find the
closest training pair by Hamming distance and predict its score s = 2^(-t0 / t_half). Every
training row at the minimum distance counts, and their scores are averaged.

Distance has two parts: allele distance (mismatches between the 34-aa pseudosequences) and
peptide distance (mismatches between the 9-mers). They are combined three ways:

  joint               allele + peptide distance, weighted equally
  allele then peptide closest allele(s) first; peptide distance only breaks ties among them
  peptide then allele closest peptide(s) first; allele distance only breaks ties among them

Same split, target and metric as baseline.py (mean per-allele Spearman on the held-out alleles).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from baseline import (AA_TO_IDX, evaluate, peptide_only_null, split_by_supertype,
                      transform_target)

# (allele weight, peptide weight). 100 exceeds the largest distance of the other part
# (34 or 9), so the heavier part decides and the lighter one only breaks ties.
WEIGHTINGS = {
    "joint (allele + peptide)": (1, 1),
    "allele first, then peptide": (100, 1),
    "peptide first, then allele": (1, 100),
}


def onehot(seqs):
    idx = np.array([[AA_TO_IDX[a] for a in s] for s in seqs])
    return np.eye(20, dtype=np.float32)[idx].reshape(len(seqs), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--chunk", type=int, default=400, help="Test rows compared at a time")
    p.add_argument("--output_dir", default="outputs/nearest_neighbour")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    s_tr = transform_target(train_df["thalf_hours"].values)
    y_te = transform_target(test_df["thalf_hours"].values)
    alleles, thalf = test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy()

    P_tr, P_te = onehot(train_df["peptide"]), onehot(test_df["peptide"])
    H_tr, H_te = onehot(train_df["hla_pseudoseq"]), onehot(test_df["hla_pseudoseq"])

    n = len(test_df)
    preds = {k: np.empty(n) for k in WEIGHTINGS}
    diag = {k: {"exact_peptide": np.empty(n), "allele_dist": np.empty(n), "n_tied": np.empty(n)}
            for k in WEIGHTINGS}
    for start in range(0, n, args.chunk):
        sl = slice(start, min(start + args.chunk, n))
        dp = 9 - P_te[sl] @ P_tr.T      # exact in float32: integer-valued, at most 34
        da = 34 - H_te[sl] @ H_tr.T
        for name, (wa, wp) in WEIGHTINGS.items():
            key = wa * da + wp * dp
            m = key == key.min(axis=1, keepdims=True)
            cnt = m.sum(axis=1)
            preds[name][sl] = (m * s_tr[None, :]).sum(axis=1) / cnt
            diag[name]["exact_peptide"][sl] = np.where(m, dp, 99).min(axis=1) == 0
            diag[name]["allele_dist"][sl] = (m * da).sum(axis=1) / cnt
            diag[name]["n_tied"][sl] = cnt

    # nearest training allele by pseudosequence, to split held-out alleles into near / far
    ps = dict(zip(df["allele"], df["hla_pseudoseq"]))
    train_alleles = sorted(train_df["allele"].unique())
    ham = lambda a, b: sum(x != y for x, y in zip(a, b))
    min_ham = {a: min(ham(ps[a], ps[b]) for b in train_alleles) for a in set(alleles)}
    near = [a for a, h in min_ham.items() if h <= 2]
    far = [a for a, h in min_ham.items() if h >= 4]

    results, per_allele = [], {}

    def record(name, pred, extra=None):
        m, per = evaluate(alleles, thalf, y_te, pred)
        per = per.set_index("allele")["scc"]
        per_allele[name] = per
        res = {"model": name, "scc": m["mean_allele_scc"], "scc_near": float(per.reindex(near).mean()),
               "scc_far": float(per.reindex(far).mean()), "n_alleles": m["n_alleles_scored"]}
        res.update(extra or {})
        results.append(res)

    record("peptide-mean null (reference)", peptide_only_null(train_df, test_df, s_tr))
    for name in WEIGHTINGS:
        d = diag[name]
        record(f"NN: {name}", preds[name],
               {"frac_exact_peptide_match": float(d["exact_peptide"].mean()),
                "mean_allele_dist_of_match": float(d["allele_dist"].mean()),
                "median_n_tied_neighbours": float(np.median(d["n_tied"]))})

    out = pd.DataFrame(results)
    pd.set_option("display.width", 220)
    print(f"\nnear = nearest training allele <=2 mismatches (n={len(near)}); "
          f"far = >=4 (n={len(far)})\n")
    print(out.round(3).to_string(index=False))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(per_allele).to_csv(os.path.join(args.output_dir, "allele_scc.csv"))
    print(f"\n[*] Wrote summary.json and allele_scc.csv to {args.output_dir}/")


if __name__ == "__main__":
    main()
