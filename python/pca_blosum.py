#!/usr/bin/env python3
"""
PCA-compressed BLOSUM50 as the per-residue encoding.

Each amino acid is a row of the (BLOSUM50 / 5) matrix, i.e. a point in R^20. PCA on those 20
points gives the k orthogonal directions of greatest variance; each residue is encoded by its
k scores. The PCA sees only the substitution matrix, never the dataset, so nothing leaks.

Same split, target, metric, network and optimiser as baseline.py, HLA input = 34-aa
pseudosequence, via the logged training loop in reduced_alphabet.py. Reference rows are the
full BLOSUM50 (k = 20) and one-hot. Train and held-out per-allele SCC and MSE are logged
during training to see whether compression narrows the train/held-out gap.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from baseline import BLOSUM50_MATRIX, split_by_supertype, transform_target
from reduced_alphabet import TABLES, encode, train_logged


def pca_table(k):
    centred = BLOSUM50_MATRIX - BLOSUM50_MATRIX.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    explained = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return (centred @ vt[:k].T).astype(np.float32), explained


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 7, 9, 12])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/pca_blosum")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tables = {f"pca{k}": pca_table(k) for k in args.ks}
    tables = {n: (t, ev) for n, (t, ev) in tables.items()}
    tables["blosum50"] = (BLOSUM50_MATRIX, 1.0)
    tables["onehot"] = (TABLES["onehot"], None)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    y_tr = transform_target(train_df["thalf_hours"].values)
    y_te = transform_target(test_df["thalf_hours"].values)
    te_alleles, te_thalf = test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy()
    tr_alleles = train_df["allele"].to_numpy()

    summary, curves = [], []
    for name, (table, explained) in tables.items():
        X_tr, X_te = encode(train_df, table), encode(test_df, table)
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
               "train_scc": float(final["train_scc"].mean()),
               "train_mse": float(final["train_mse"].mean()),
               "test_mse": float(final["test_mse"].mean()),
               "peak_scc_seed_avg": float(mean["scc"].max()),
               "peak_epoch": int(mean["scc"].idxmax())}
        res["scc_gap"] = res["train_scc"] - res["scc_mean"]
        res["mse_gap"] = res["test_mse"] - res["train_mse"]
        summary.append(res)
        ve = "" if explained is None else f"var {explained:.0%} | "
        print(f"{name:9s} dim {res['input_dim']:4d} params {res['n_params']:6d} | {ve}"
              f"test SCC {res['scc_mean']:.3f} +/- {res['scc_sd']:.3f} | train SCC "
              f"{res['train_scc']:.3f} | SCC gap {res['scc_gap']:.3f} | MSE gap "
              f"{res['mse_gap']:.3f} | peak {res['peak_scc_seed_avg']:.3f}@{res['peak_epoch']}",
              flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "variants": summary}, f, indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)


if __name__ == "__main__":
    main()
