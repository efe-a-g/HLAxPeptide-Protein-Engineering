#!/usr/bin/env python3
"""
Reduced amino-acid encodings vs one-hot, with an overfitting diagnostic.

Same split, target, metric, network (StabilityNet, 60 sigmoid units) and optimiser as
baseline.py; HLA input is the 34-aa pseudosequence. Only the per-residue encoding changes:

  onehot    NetMHCpan sparse encoding, 20 dims/residue (reference)
  group5    5 physicochemical classes, 5 dims/residue (one-hot over classes)
  sidney14  multi-hot over the 14 pocket-specificity groups of Sidney et al. 2008 Table 1,
            14 dims/residue; residues in no group (C, G, N) encode as all zeros
  both      group5 and sidney14 concatenated, 19 dims/residue

The training loop is re-implemented here (same Adam / ReduceLROnPlateau / 90 epochs) so
that train MSE, held-out MSE and held-out per-allele SCC can be logged during training.
The held-out curves are for diagnosis only; the reported score is the final epoch, as in
the baseline, so no epoch is chosen using the test alleles.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (AA_ORDER, AA_TO_IDX, StabilityNet, evaluate, resolve_device,
                      split_by_supertype, transform_target)

# Standard 5-class physicochemical grouping.
GROUP5 = ["GAVLIMP", "FWY", "STCNQ", "KRH", "DE"]  # nonpolar aliphatic, aromatic, polar, +, -

# Sidney et al. 2008, BMC Immunol 9:1, Table 1 (as read from the PMC page).
SIDNEY14 = ["DE", "RHK", "FWY", "LIVMQ", "AST", "AV", "FLIM", "P", "LIVMFWYA",
            "FWYLIVMQ", "FWYLIM", "YRK", "ATSVLIMQ", "ASTVLIMQFWY"]


def residue_table(groups, partition):
    """(20, len(groups)) matrix; row i is the encoding of AA_ORDER[i]."""
    m = np.zeros((20, len(groups)), dtype=np.float32)
    for j, g in enumerate(groups):
        for aa in g:
            m[AA_TO_IDX[aa], j] = 1.0
    if partition:
        assert (m.sum(1) == 1).all(), "groups must partition the 20 amino acids"
    return m


TABLES = {
    "onehot": np.eye(20, dtype=np.float32) * 0.85 + 0.05,
    "group5": residue_table(GROUP5, partition=True),
    "sidney14": residue_table(SIDNEY14, partition=False),
}
TABLES["both"] = np.hstack([TABLES["group5"], TABLES["sidney14"]])


def encode(df, table):
    seqs = [p + h for p, h in zip(df["peptide"], df["hla_pseudoseq"])]
    idx = np.array([[AA_TO_IDX[a] for a in s] for s in seqs], dtype=np.int32)
    return table[idx].reshape(len(seqs), -1).astype(np.float32)


def train_logged(X_tr, y_tr, X_te, y_te, test_alleles, test_thalf, args, eval_every,
                 train_alleles=None):
    """If `train_alleles` is given, also log per-allele SCC on the training rows."""
    device = resolve_device(args.device)
    model = StabilityNet(X_tr.shape[1], args.hidden_dim).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=args.batch_size, shuffle=True)
    Xte = torch.from_numpy(X_te).to(device)
    Xtr_t = torch.from_numpy(X_tr).to(device) if train_alleles is not None else None

    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(bx)
        total /= len(y_tr)
        sched.step(total)
        if epoch % eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred = model(Xte).cpu().numpy()
            m, _ = evaluate(test_alleles, test_thalf, y_te, pred)
            row = {"epoch": epoch, "train_mse": total,
                   "test_mse": float(np.mean((pred - y_te) ** 2)),
                   "scc": m["mean_allele_scc"]}
            if train_alleles is not None:
                with torch.no_grad():
                    tr_pred = model(Xtr_t).cpu().numpy()
                row["train_scc"] = evaluate(train_alleles, None, y_tr, tr_pred)[0]["mean_allele_scc"]
            curve.append(row)
    return curve, sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--encodings", nargs="+", default=["onehot", "group5", "sidney14", "both"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/reduced_alphabet")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    y_tr = transform_target(train_df["thalf_hours"].values)
    y_te = transform_target(test_df["thalf_hours"].values)
    alleles, thalf = test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy()

    summary, curves = [], []
    for name in args.encodings:
        X_tr, X_te = encode(train_df, TABLES[name]), encode(test_df, TABLES[name])
        runs = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            curve, n_params = train_logged(X_tr, y_tr, X_te, y_te, alleles, thalf, args,
                                           args.eval_every)
            for row in curve:
                curves.append({"encoding": name, "seed": seed, **row})
            runs.append(pd.DataFrame(curve).set_index("epoch"))
        mean = sum(runs) / len(runs)  # seed-averaged curve
        final = [r.iloc[-1] for r in runs]
        peak_epoch = int(mean["scc"].idxmax())
        res = {
            "encoding": name, "input_dim": int(X_tr.shape[1]), "n_params": int(n_params),
            "final_scc_mean": float(np.mean([f["scc"] for f in final])),
            "final_scc_sd": float(np.std([f["scc"] for f in final])),
            "final_scc_per_seed": [float(f["scc"]) for f in final],
            "final_train_mse": float(np.mean([f["train_mse"] for f in final])),
            "final_test_mse": float(np.mean([f["test_mse"] for f in final])),
            "peak_scc_seed_avg": float(mean["scc"].max()), "peak_epoch": peak_epoch,
            "min_test_mse_epoch": int(mean["test_mse"].idxmin()),
        }
        res["mse_gap"] = res["final_test_mse"] - res["final_train_mse"]
        summary.append(res)
        print(f"{name:9s} dim {res['input_dim']:4d} params {res['n_params']:6d} | "
              f"final SCC {res['final_scc_mean']:.3f} +/- {res['final_scc_sd']:.3f} | "
              f"peak {res['peak_scc_seed_avg']:.3f}@{peak_epoch} | "
              f"train MSE {res['final_train_mse']:.4f} test MSE {res['final_test_mse']:.4f} "
              f"(min test MSE @ {res['min_test_mse_epoch']})", flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "group5": GROUP5, "sidney14": SIDNEY14,
                   "variants": summary}, f, indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)


if __name__ == "__main__":
    main()
