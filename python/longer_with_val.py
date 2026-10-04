#!/usr/bin/env python3
"""
How long should this train, and can we tell without looking at the test alleles?

baseline.py trains a fixed 90 epochs with no validation set, and every curve logged so far
peaks at epoch 90 -- the last one -- so the budget is binding and the true optimum is unknown.
The foundation-models branch measured +0.093 mean-allele PCC from 25 -> 100 epochs on its
older splits and listed "never re-checked on the canonical split" as an open gap.

Protocol here, which also fills the repo's missing-validation-set gap:

  outer  split_by_supertype(df, seed=42)                -> train / test   (test untouched)
  inner  split_by_supertype(train_df, seed=--val_seed)  -> fit / val      (held-out ALLELES)

The inner split holds out whole alleles too, so selecting on it selects for the thing we
actually want -- generalisation to an unseen allele -- rather than to unseen peptides.
Models train on `fit` only; the epoch is chosen by val SCC; test is read once at that epoch.

Reported per variant:
  val-selected   test SCC at the epoch with the best val SCC      <- the honest number
  at 90          test SCC at epoch 90                             <- baseline.py's budget
  oracle peak    best test SCC over all epochs                    <- NOT selectable, context
                                                                     only: the gap between it
                                                                     and val-selected is what
                                                                     early stopping costs.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (StabilityNet, encode_sequences, evaluate, resolve_device,
                      split_by_supertype, transform_target)
from ranking_loss import LinearOutNet
from target_transform import make_target, within_allele_rank


def features(d):
    return encode_sequences(d["peptide"].values, d["hla_pseudoseq"].values, encoding="sparse")


def train_curve(X_fit, y_fit, evals, args, seed, sigmoid):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    model = (StabilityNet(X_fit.shape[1], args.hidden_dim) if sigmoid
             else LinearOutNet(X_fit.shape[1], args.hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=args.patience)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit)),
                        batch_size=args.batch_size, shuffle=True)
    tensors = {k: torch.from_numpy(v["X"]).to(device) for k, v in evals.items()}

    curve, preds_at = [], {}
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
            row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"]}
            with torch.no_grad():
                for k, v in evals.items():
                    pred = model(tensors[k]).cpu().numpy()
                    row[f"{k}_scc"] = evaluate(v["alleles"], None, v["s"],
                                               pred)[0]["mean_allele_scc"]
                    if k == "test":
                        preds_at[epoch] = pred
            curve.append(row)
    return pd.DataFrame(curve).set_index("epoch"), preds_at


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--targets", nargs="+", default=["raw", "centered"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--val_seed", type=int, default=7, help="seed for the inner fit/val split")
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/longer_with_val")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos].reset_index(drop=True), df.iloc[test_pos]
    print("[*] inner fit/val split (held-out alleles within train):")
    fit_pos, val_pos = split_by_supertype(train_df, seed=args.val_seed)
    fit_df, val_df = train_df.iloc[fit_pos], train_df.iloc[val_pos]

    s_fit = transform_target(fit_df["thalf_hours"].values)
    evals = {
        "fit":  {"X": features(fit_df),  "alleles": fit_df["allele"].to_numpy(),  "s": s_fit},
        "val":  {"X": features(val_df),  "alleles": val_df["allele"].to_numpy(),
                 "s": transform_target(val_df["thalf_hours"].values)},
        "test": {"X": features(test_df), "alleles": test_df["allele"].to_numpy(),
                 "s": transform_target(test_df["thalf_hours"].values)},
    }
    X_fit = evals["fit"]["X"]
    fit_allele_ids = pd.factorize(fit_df["allele"])[0]
    te_alleles, s_te = evals["test"]["alleles"], evals["test"]["s"]
    print(f"[*] fit {len(fit_df):,} rows / val {len(val_df):,} rows "
          f"({val_df['allele'].nunique()} val alleles) / test {len(test_df):,} rows\n")

    summary, all_curves = [], []
    for name in args.targets:
        y_fit = make_target(name, s_fit, fit_allele_ids)
        curves, preds_by_epoch = [], []
        for seed in args.seeds:
            c, preds_at = train_curve(X_fit, y_fit, evals, args, seed,
                                      sigmoid=(name == "raw"))
            curves.append(c)
            preds_by_epoch.append(preds_at)
            all_curves += [{"target": name, "seed": seed, "epoch": e, **r}
                           for e, r in c.iterrows()]
        mean = sum(curves) / len(curves)
        best_val_epoch = int(mean["val_scc"].idxmax())
        at90 = int(min(mean.index, key=lambda e: abs(e - 90)))

        def ens(epoch):
            P = np.stack([pa[epoch] for pa in preds_by_epoch])
            return (float(evaluate(te_alleles, None, s_te, P.mean(0))[0]["mean_allele_scc"]),
                    float(evaluate(te_alleles, None, s_te,
                                   np.stack([within_allele_rank(q, te_alleles) for q in P]
                                            ).mean(0))[0]["mean_allele_scc"]))

        ens_sel, ens_sel_rank = ens(best_val_epoch)
        res = {"target": name, "best_val_epoch": best_val_epoch,
               "val_scc_at_best": float(mean.loc[best_val_epoch, "val_scc"]),
               "test_at_val_selected": float(mean.loc[best_val_epoch, "test_scc"]),
               "test_at_90": float(mean.loc[at90, "test_scc"]), "epoch_used_for_90": at90,
               "oracle_peak_test": float(mean["test_scc"].max()),
               "oracle_peak_epoch": int(mean["test_scc"].idxmax()),
               "fit_scc_at_selected": float(mean.loc[best_val_epoch, "fit_scc"]),
               "lr_at_selected": float(mean.loc[best_val_epoch, "lr"]),
               "ensemble_at_val_selected": ens_sel,
               "ensemble_rank_at_val_selected": ens_sel_rank,
               "ensemble_at_90": ens(at90)[0]}
        summary.append(res)
        print(f"{name:9s} val picks epoch {best_val_epoch:3d} (val {res['val_scc_at_best']:.4f}) "
              f"-> test {res['test_at_val_selected']:.4f} | ens {ens_sel:.4f} || "
              f"test@{at90} {res['test_at_90']:.4f} (ens {res['ensemble_at_90']:.4f}) || "
              f"oracle {res['oracle_peak_test']:.4f}@{res['oracle_peak_epoch']} | "
              f"fit {res['fit_scc_at_selected']:.3f} | lr {res['lr_at_selected']:.2e}",
              flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "val_seed": args.val_seed, "epochs": args.epochs,
                   "variants": summary}, f, indent=2)
    pd.DataFrame(all_curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)
    print(f"\n[*] Wrote summary.json and curves.csv to {args.output_dir}/")


if __name__ == "__main__":
    main()
