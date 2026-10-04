#!/usr/bin/env python3
"""
Encoding ablation on the project-wide split from baseline.py.

  Peptide encoding   : blosum50 vs sparse (one-hot)
  HLA input          : 34-aa pseudosequence vs full 182-aa sequence
Crossed into 2x2 (the same encoding is used for peptide and HLA), each trained with
several seeds using the baseline network and training loop.
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import torch

from baseline import (AA_TO_IDX, BLOSUM50_MATRIX, evaluate, split_by_supertype,
                      train_model, transform_target)


def encode(seqs, encoding):
    idx = np.array([[AA_TO_IDX[a] for a in s] for s in seqs], dtype=np.int32)
    if encoding == "blosum50":
        f = BLOSUM50_MATRIX[idx]
    else:
        f = np.eye(20, dtype=np.float32)[idx] * 0.85 + 0.05
    return f.reshape(len(seqs), -1).astype(np.float32)


def build(df, encoding, hla_col):
    # HLA features are per-allele; encode unique sequences once and broadcast.
    pep = encode(df["peptide"].values, encoding)
    uniq = df[hla_col].unique()
    table = dict(zip(uniq, encode(uniq, encoding)))
    hla = np.stack([table[s] for s in df[hla_col].values])
    return np.hstack([pep, hla])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/encoding")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    y_train = transform_target(train_df["thalf_hours"].values)
    y_test = transform_target(test_df["thalf_hours"].values)

    results, per_allele = [], []
    for hla_col in ("hla_pseudoseq", "hla_seq"):
        for encoding in ("sparse", "blosum50"):
            X_train = build(train_df, encoding, hla_col)
            X_test = build(test_df, encoding, hla_col)
            name = f"{encoding} + {'pseudoseq' if hla_col == 'hla_pseudoseq' else 'full seq'}"
            scores, pers = [], []
            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    _, predict = train_model(X_train, y_train, args)
                m, per = evaluate(test_df["allele"].to_numpy(),
                                  test_df["thalf_hours"].to_numpy(), y_test, predict(X_test))
                scores.append(m["mean_allele_scc"])
                pers.append(per.set_index("allele")["scc"])
            print(f"{name:26s} dim {X_train.shape[1]:5d}  SCC {np.mean(scores):.3f} "
                  f"+/- {np.std(scores):.3f}  ({', '.join(f'{s:.3f}' for s in scores)})",
                  flush=True)
            results.append({"name": name, "dim": int(X_train.shape[1]),
                            "scc_mean": float(np.mean(scores)), "scc_sd": float(np.std(scores)),
                            "scc_per_seed": [float(s) for s in scores]})
            per_allele.append(pd.concat(pers, axis=1).mean(axis=1).rename(name))

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "variants": results}, f, indent=2)
    pd.concat(per_allele, axis=1).to_csv(os.path.join(args.output_dir, "allele_scc.csv"))


if __name__ == "__main__":
    main()
