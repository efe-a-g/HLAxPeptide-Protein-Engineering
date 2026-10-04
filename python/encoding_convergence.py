#!/usr/bin/env python3
"""
Re-run every earlier input-compression experiment at convergence instead of 90 epochs.

All of them -- reduced alphabets, PCA-compressed BLOSUM50, pocket-position masking, full
sequence -- were measured at baseline.py's fixed 90-epoch budget. That budget turned out to be
well short of convergence (longer_with_val.py; final_recipe.py), and a truncated budget
flatters whatever optimises fastest. Smaller inputs optimise faster, so the earlier numbers
were biased in favour of compression on the speed axis and against it on the accuracy axis,
and neither bias can be signed without re-running.

Protocol is final_recipe.py's, so nothing is chosen on the test alleles:
  stage 1  inner fit/val split of train (whole alleles held out); val picks the epoch
  stage 2  refit on all of train for that many epochs; read test once; ensemble over seeds

Architecture and target are baseline.py's exactly (one 60-unit sigmoid layer, sigmoid output,
MSE on s), so the only thing varying is the encoding -- the comparison the earlier runs meant
to make. Reports held-out SCC *and* the train/held-out gap, since the question is whether
compression buys generalisation or merely buys a smaller gap by underfitting.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (AA_TO_IDX, BLOSUM50_MATRIX, StabilityNet, evaluate, resolve_device,
                      split_by_supertype, transform_target)
from masked_positions import HLA_POS, PEP_POS
from pca_blosum import pca_table
from reduced_alphabet import TABLES as REDUCED_TABLES
from target_transform import within_allele_rank

ONEHOT = np.eye(20, dtype=np.float32) * 0.85 + 0.05


def residue_table(name):
    """Per-residue encoding matrix (20, d) for the table-based encodings."""
    if name == "onehot":
        return ONEHOT
    if name == "blosum50":
        return BLOSUM50_MATRIX
    if name in REDUCED_TABLES:
        return REDUCED_TABLES[name]
    if name.startswith("pca"):
        return pca_table(int(name[3:]))[0]
    return None


def encode(df, spec):
    """spec is <table> (peptide + 34-aa pseudoseq), or <table>@full / <table>@masked."""
    table_name, _, variant = spec.partition("@")
    table = residue_table(table_name)
    if table is None:
        raise ValueError(f"unknown encoding {spec}")
    if variant == "masked":
        idx = np.array([[AA_TO_IDX[p[i - 1]] for i in PEP_POS]
                        + [AA_TO_IDX[h[i - 1]] for i in HLA_POS]
                        for p, h in zip(df["peptide"], df["hla_seq"])], dtype=np.int32)
    else:
        hla_col = "hla_seq" if variant == "full" else "hla_pseudoseq"
        idx = np.array([[AA_TO_IDX[a] for a in p + h]
                        for p, h in zip(df["peptide"], df[hla_col])], dtype=np.int32)
    return table[idx].reshape(len(df), -1).astype(np.float32)


def run(X_fit, y_fit, eval_sets, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    model = StabilityNet(X_fit.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit)),
                        batch_size=args.batch_size, shuffle=True)
    tens = {k: torch.from_numpy(v["X"]).to(device) for k, v in eval_sets.items()}

    curve, stored = [], {}
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
        sched.step(total / len(y_fit))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row = {"epoch": epoch}
            with torch.no_grad():
                for k, v in eval_sets.items():
                    pred = model(tens[k]).cpu().numpy()
                    row[f"{k}_scc"] = evaluate(v["alleles"], None, v["s"],
                                               pred)[0]["mean_allele_scc"]
                    if k == "test":
                        stored[epoch] = pred
            curve.append(row)
    return pd.DataFrame(curve).set_index("epoch"), stored


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--encodings", nargs="+",
                   default=["onehot", "blosum50", "group5", "sidney14", "both",
                            "pca5", "pca9", "pca12", "onehot@masked", "onehot@full"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--select_seeds", type=int, nargs="+", default=[42, 1])
    p.add_argument("--val_seed", type=int, default=7)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/encoding_convergence")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df = df.iloc[train_pos].reset_index(drop=True)
    test_df = df.iloc[test_pos]
    print("[*] inner fit/val split within train:")
    fit_pos, val_pos = split_by_supertype(train_df, seed=args.val_seed)
    fit_df, val_df = train_df.iloc[fit_pos], train_df.iloc[val_pos]

    frames = {"fit": fit_df, "val": val_df, "train": train_df, "test": test_df}
    s = {k: transform_target(d["thalf_hours"].values) for k, d in frames.items()}
    alleles = {k: d["allele"].to_numpy() for k, d in frames.items()}
    print()

    summary, curves = [], []
    for spec in args.encodings:
        X = {k: encode(d, spec) for k, d in frames.items()}
        sets1 = {k: {"X": X[k], "alleles": alleles[k], "s": s[k]} for k in ("fit", "val", "test")}
        sets2 = {k: {"X": X[k], "alleles": alleles[k], "s": s[k]} for k in ("train", "test")}

        sel = []
        for seed in args.select_seeds:
            c, _ = run(X["fit"], s["fit"], sets1, args, seed)
            sel.append(c)
            curves += [{"encoding": spec, "stage": 1, "seed": seed, "epoch": e, **r}
                       for e, r in c.iterrows()]
        sel = sum(sel) / len(sel)
        best_epoch = int(sel["val_scc"].idxmax())

        ep_backup, args.epochs = args.epochs, best_epoch
        preds, per_seed, train_sccs = [], [], []
        for seed in args.seeds:
            c, stored = run(X["train"], s["train"], sets2, args, seed)
            preds.append(stored[best_epoch])
            per_seed.append(evaluate(alleles["test"], None, s["test"],
                                     stored[best_epoch])[0]["mean_allele_scc"])
            train_sccs.append(c.loc[best_epoch, "train_scc"])
            curves += [{"encoding": spec, "stage": 2, "seed": seed, "epoch": e, **r}
                       for e, r in c.iterrows()]
        args.epochs = ep_backup
        preds = np.stack(preds)

        res = {"encoding": spec, "input_dim": int(X["train"].shape[1]),
               "n_params": int(60 * X["train"].shape[1] + 60 + 61),
               "selected_epoch": best_epoch,
               "mean_of_seeds": float(np.mean(per_seed)), "sd_of_seeds": float(np.std(per_seed)),
               "per_seed": [float(x) for x in per_seed],
               "ensemble": float(evaluate(alleles["test"], None, s["test"],
                                          preds.mean(0))[0]["mean_allele_scc"]),
               "train_scc": float(np.mean(train_sccs)),
               "stage1_oracle_peak": float(sel["test_scc"].max()),
               "stage1_oracle_epoch": int(sel["test_scc"].idxmax())}
        res["gap"] = res["train_scc"] - res["mean_of_seeds"]
        summary.append(res)
        np.save(os.path.join(args.output_dir, f"preds_{spec.replace('@', '_')}.npy"), preds)
        print(f"{spec:16s} dim {res['input_dim']:5d} ep {best_epoch:3d} | "
              f"test {res['mean_of_seeds']:.4f} +/- {res['sd_of_seeds']:.4f} | "
              f"ENS {res['ensemble']:.4f} | train {res['train_scc']:.3f} | "
              f"GAP {res['gap']:.3f}", flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "select_seeds": args.select_seeds,
                   "val_seed": args.val_seed, "max_epochs": args.epochs,
                   "variants": summary}, f, indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)
    np.save(os.path.join(args.output_dir, "test_alleles.npy"), alleles["test"])
    np.save(os.path.join(args.output_dir, "test_s.npy"), s["test"])
    print(f"\n[*] Wrote to {args.output_dir}/")


if __name__ == "__main__":
    main()
