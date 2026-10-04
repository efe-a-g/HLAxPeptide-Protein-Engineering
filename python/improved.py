#!/usr/bin/env python3
"""
Two experiments on top of the baseline, both on the project-wide split from baseline.py.

  1. Deeper network: 2 or 3 hidden layers with dropout, instead of one sigmoid layer.
  2. Allele-level log10(TD) appended to the input. Only alleles with a TD value in
     data/TD_features.csv are used, in both train and test; the plain baseline network is
     retrained on the same subset so the comparison is like-for-like.

Each variant is trained with several seeds, because the baseline's mean per-allele SCC
moves by about +/-0.02 from seed to seed.
"""

import argparse
import contextlib
import io
import json
import os
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baseline import (encode_sequences, evaluate, peptide_only_null, split_by_supertype,
                      train_model, transform_target)


class DeepStabilityNet(nn.Module):
    """`n_hidden` layers of ReLU + dropout, sigmoid output."""

    def __init__(self, in_features, n_hidden=2, hidden_dim=60, dropout=0.2):
        super().__init__()
        layers, d = [], in_features
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net(x)).squeeze(-1)


def run_variant(name, make_model, X_train, y_train, X_test, test_df, y_test, seeds, args):
    """Train once per seed; return per-seed headline scores and seed-averaged per-allele SCC."""
    scores, per_allele = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            _, predict_fn = train_model(X_train, y_train, args, model=make_model(X_train.shape[1]))
        metrics, per = evaluate(test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy(),
                                y_test, predict_fn(X_test))
        scores.append(metrics["mean_allele_scc"])
        per_allele.append(per.set_index("allele")["scc"])
    per_allele = pd.concat(per_allele, axis=1).mean(axis=1).rename(name)
    print(f"    {name:42s} SCC {np.mean(scores):.3f} +/- {np.std(scores):.3f}   "
          f"(seeds: {', '.join(f'{s:.3f}' for s in scores)})")
    return {"name": name, "scc_mean": float(np.mean(scores)), "scc_sd": float(np.std(scores)),
            "scc_per_seed": [float(s) for s in scores]}, per_allele


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--td_path", default="data/TD_features.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/improved")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]

    def features(d):
        return encode_sequences(d["peptide"].values, d["hla_pseudoseq"].values)

    X_train, X_test = features(train_df), features(test_df)
    y_train = transform_target(train_df["thalf_hours"].values)
    y_test = transform_target(test_df["thalf_hours"].values)

    results, per_allele = [], []

    def record(res, per):
        results.append(res)
        per_allele.append(per)

    print("\n[1] Depth and dropout (all 16 held-out alleles)")
    record(*run_variant("baseline: 1 hidden layer, sigmoid",
                        lambda d: None, X_train, y_train, X_test, test_df, y_test,
                        args.seeds, args))
    for n_hidden in (2, 3):
        record(*run_variant(
            f"{n_hidden} hidden layers + dropout {args.dropout}",
            lambda d, n=n_hidden: DeepStabilityNet(d, n, args.hidden_dim, args.dropout),
            X_train, y_train, X_test, test_df, y_test, args.seeds, args))

    print("\n[2] log10(TD) as an extra input (alleles with a TD value only)")
    td = pd.read_csv(args.td_path).set_index("allele")
    has_td = set(td.index[td["TD_available"]])
    tr_td = train_df[train_df["allele"].isin(has_td)]
    te_td = test_df[test_df["allele"].isin(has_td)]
    dropped = sorted(set(test_df["allele"]) - set(te_td["allele"]))
    print(f"    train {len(train_df):,} -> {len(tr_td):,} rows | test alleles "
          f"{test_df['allele'].nunique()} -> {te_td['allele'].nunique()}; "
          f"dropped from test (no TD): {', '.join(dropped)}")

    # Standardise with train statistics; one value per allele, so it is constant within an allele.
    mu, sd = td.loc[tr_td["allele"].unique(), "log10_TD"].agg(["mean", "std"])

    def with_td(d):
        col = ((d["allele"].map(td["log10_TD"]).to_numpy() - mu) / sd).astype(np.float32)
        return np.hstack([features(d), col[:, None]])

    Xtr_td, Xte_td = with_td(tr_td), with_td(te_td)
    ytr_td = transform_target(tr_td["thalf_hours"].values)
    yte_td = transform_target(te_td["thalf_hours"].values)
    null, _ = evaluate(te_td["allele"].to_numpy(), te_td["thalf_hours"].to_numpy(), yte_td,
                       peptide_only_null(tr_td, te_td, ytr_td))
    print(f"    peptide-only null on this subset: {null['mean_allele_scc']:.3f}")
    record(*run_variant("TD subset: baseline net, no TD",
                        lambda d: None, Xtr_td[:, :-1], ytr_td, Xte_td[:, :-1], te_td, yte_td,
                        args.seeds, args))
    record(*run_variant("TD subset: baseline net + log10_TD",
                        lambda d: None, Xtr_td, ytr_td, Xte_td, te_td, yte_td, args.seeds, args))

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "td_subset_null_scc": null["mean_allele_scc"],
                   "td_subset_dropped_test_alleles": dropped, "variants": results}, f, indent=2)
    pd.concat(per_allele, axis=1).to_csv(os.path.join(args.output_dir, "allele_scc.csv"))
    print(f"\n[*] Wrote summary.json and allele_scc.csv to {args.output_dir}/")


if __name__ == "__main__":
    main()
