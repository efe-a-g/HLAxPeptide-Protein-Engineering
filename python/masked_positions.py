#!/usr/bin/env python3
"""
Position-masked inputs: peptide P2 and P9 plus 20 HLA pocket residues (22 positions).

  B pocket (binds P2) : 7 9 24 45 63 66 67 70 99
  F pocket (binds P9) : 74 77 80 81 84 95 97 114 116 143 147
HLA numbers are 1-based residue positions in `hla_seq`; all 20 lie inside the 34-position
NetMHCpan pseudosequence. Everything else (split, target, metric, network, optimiser,
logging) is the same as in pca_blosum.py; only which positions enter the input changes.

Encodings: onehot, blosum50, pca<k> (BLOSUM50 compressed to k principal components).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from baseline import AA_TO_IDX, BLOSUM50_MATRIX, split_by_supertype, transform_target
from pca_blosum import pca_table
from reduced_alphabet import TABLES, train_logged

PEP_POS = [2, 9]                                    # 1-based peptide positions
B_POCKET = [7, 9, 24, 45, 63, 66, 67, 70, 99]
F_POCKET = [74, 77, 80, 81, 84, 95, 97, 114, 116, 143, 147]
HLA_POS = B_POCKET + F_POCKET


def encode_masked(df, table):
    idx = np.array([[AA_TO_IDX[p[i - 1]] for i in PEP_POS] + [AA_TO_IDX[h[i - 1]] for i in HLA_POS]
                    for p, h in zip(df["peptide"], df["hla_seq"])], dtype=np.int32)
    return table[idx].reshape(len(df), -1).astype(np.float32)


def get_table(name):
    if name == "onehot":
        return TABLES["onehot"], None
    if name == "blosum50":
        return BLOSUM50_MATRIX, 1.0
    return pca_table(int(name[3:]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--encodings", nargs="+", default=["onehot", "blosum50", "pca7", "pca9", "pca12"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/masked_positions")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    y_tr = transform_target(train_df["thalf_hours"].values)
    y_te = transform_target(test_df["thalf_hours"].values)
    te_alleles, te_thalf = test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy()
    tr_alleles = train_df["allele"].to_numpy()

    summary, curves = [], []
    for name in args.encodings:
        table, explained = get_table(name)
        X_tr, X_te = encode_masked(train_df, table), encode_masked(test_df, table)
        runs = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            curve, n_params = train_logged(X_tr, y_tr, X_te, y_te, te_alleles, te_thalf, args,
                                           args.eval_every, train_alleles=tr_alleles)
            curves += [{"encoding": name, "seed": seed, **r} for r in curve]
            runs.append(pd.DataFrame(curve).set_index("epoch"))
        mean = sum(runs) / len(runs)
        final = pd.DataFrame([r.iloc[-1] for r in runs])
        res = {"encoding": name, "input_dim": int(X_tr.shape[1]), "n_params": int(n_params),
               "variance_explained": explained,
               "scc_mean": float(final["scc"].mean()), "scc_sd": float(final["scc"].std(ddof=0)),
               "scc_per_seed": [float(x) for x in final["scc"]],
               "train_scc": float(final["train_scc"].mean()),
               "train_mse": float(final["train_mse"].mean()),
               "test_mse": float(final["test_mse"].mean()),
               "peak_scc_seed_avg": float(mean["scc"].max()),
               "peak_epoch": int(mean["scc"].idxmax())}
        res["scc_gap"] = res["train_scc"] - res["scc_mean"]
        res["mse_gap"] = res["test_mse"] - res["train_mse"]
        summary.append(res)
        print(f"{name:9s} dim {res['input_dim']:4d} params {res['n_params']:6d} | "
              f"test SCC {res['scc_mean']:.3f} +/- {res['scc_sd']:.3f} | train SCC "
              f"{res['train_scc']:.3f} | SCC gap {res['scc_gap']:.3f} | MSE gap "
              f"{res['mse_gap']:.3f} | peak {res['peak_scc_seed_avg']:.3f}@{res['peak_epoch']}",
              flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "pep_pos": PEP_POS, "hla_pos": HLA_POS,
                   "variants": summary}, f, indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)


if __name__ == "__main__":
    main()
