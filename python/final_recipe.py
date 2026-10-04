#!/usr/bin/env python3
"""
Combine the three independent gains, with the epoch chosen honestly, and refit on all of train.

Each piece was measured separately against baseline.py's 0.416 on the canonical split:

  rank target      within-allele rank -> normal quantile   (target_transform.py)
  architecture     bilinear / PSSM instead of a dense MLP  (pssm_head.py)
  epoch budget     90 epochs is well short of convergence  (longer_with_val.py)
  seed ensemble    average predictions over seeds, free    (target_transform.py)

Two-stage protocol, so no number is selected on the test alleles:

  Stage 1  outer split -> train / test, then an inner split of TRAIN into fit / val holding out
           whole alleles. Train on fit, pick the epoch E* with the best val SCC. Test is never
           read here.
  Stage 2  refit on the FULL train set for E* epochs, one model per seed, and read the test
           alleles once. Stage 1 throws away ~25% of the training alleles to build val, so the
           refit is what you would actually ship.

The test alleles are read once per configuration, at an epoch fixed before they were touched.
`oracle_peak` columns are printed for context only and are never used to choose anything.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (evaluate, peptide_only_null, resolve_device, split_by_supertype,
                      transform_target)
from pssm_head import MODELS, encode_parts
from target_transform import make_target, within_allele_rank


def run(make, parts_fit, y_fit, eval_sets, args, seed):
    """Train on `parts_fit`; return per-epoch SCC on each eval set and the stored predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    fit_t = [torch.from_numpy(a).to(device) for a in parts_fit]
    sets = {k: ([torch.from_numpy(a).to(device) for a in v["parts"]], v) for k, v in
            eval_sets.items()}
    model = make(fit_t[1].shape[1], fit_t[2].shape[1], args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=args.patience)
    crit = nn.MSELoss()
    yt = torch.from_numpy(y_fit).to(device)
    loader = DataLoader(TensorDataset(torch.arange(len(y_fit))), batch_size=args.batch_size,
                        shuffle=True)

    curve, stored = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (idx,) in loader:
            idx = idx.to(device)
            opt.zero_grad()
            loss = crit(model(fit_t[0][idx], fit_t[1][idx], fit_t[2][idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step(total / len(y_fit))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row = {"epoch": epoch}
            with torch.no_grad():
                for k, (tens, v) in sets.items():
                    pred = model(*tens).cpu().numpy()
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
    p.add_argument("--configs", nargs="+", default=["mlp:raw", "mlp:rank", "pssm:rank",
                                                    "bilinear:rank"],
                   help="model:target pairs, e.g. bilinear:rank")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--select_seeds", type=int, nargs="+", default=[42, 1],
                   help="seeds used in stage 1 to choose the epoch")
    p.add_argument("--val_seed", type=int, default=7)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/final_recipe")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df = df.iloc[train_pos].reset_index(drop=True)
    test_df = df.iloc[test_pos]
    print("[*] inner fit/val split within train:")
    fit_pos, val_pos = split_by_supertype(train_df, seed=args.val_seed)
    fit_df, val_df = train_df.iloc[fit_pos], train_df.iloc[val_pos]

    parts = {k: encode_parts(d) for k, d in
             (("fit", fit_df), ("val", val_df), ("train", train_df), ("test", test_df))}
    s = {k: transform_target(d["thalf_hours"].values) for k, d in
         (("fit", fit_df), ("val", val_df), ("train", train_df), ("test", test_df))}
    alleles = {k: d["allele"].to_numpy() for k, d in
               (("fit", fit_df), ("val", val_df), ("train", train_df), ("test", test_df))}
    ids = {k: pd.factorize(alleles[k])[0] for k in ("fit", "train")}

    stage1_sets = {k: {"parts": parts[k], "alleles": alleles[k], "s": s[k]}
                   for k in ("fit", "val", "test")}
    stage2_sets = {"test": {"parts": parts["test"], "alleles": alleles["test"], "s": s["test"]}}

    null = evaluate(alleles["test"], None, s["test"],
                    peptide_only_null(train_df, test_df, s["train"]))[0]["mean_allele_scc"]
    print(f"[*] peptide-only null {null:.4f} | baseline.py reference 0.4156\n")

    summary, curves = [], []
    for cfg in args.configs:
        model_name, target = cfg.split(":")
        make = MODELS[model_name]

        # Stage 1: choose the epoch on val alleles only.
        y_fit = make_target(target, s["fit"], ids["fit"])
        sel = []
        for seed in args.select_seeds:
            c, _ = run(make, parts["fit"], y_fit, stage1_sets, args, seed)
            sel.append(c)
            curves += [{"config": cfg, "stage": 1, "seed": seed, "epoch": e, **r}
                       for e, r in c.iterrows()]
        sel = sum(sel) / len(sel)
        best_epoch = int(sel["val_scc"].idxmax())

        # Stage 2: refit on all of train for that many epochs; read test once.
        y_train = make_target(target, s["train"], ids["train"])
        ep_backup, args.epochs = args.epochs, best_epoch
        preds, per_seed = [], []
        for seed in args.seeds:
            c, stored = run(make, parts["train"], y_train, stage2_sets, args, seed)
            pred = stored[best_epoch]
            preds.append(pred)
            per_seed.append(evaluate(alleles["test"], None, s["test"],
                                     pred)[0]["mean_allele_scc"])
            curves += [{"config": cfg, "stage": 2, "seed": seed, "epoch": e, **r}
                       for e, r in c.iterrows()]
        args.epochs = ep_backup
        preds = np.stack(preds)

        res = {"config": cfg, "model": model_name, "target": target,
               "selected_epoch": best_epoch,
               "val_scc_at_selected": float(sel.loc[best_epoch, "val_scc"]),
               "stage1_test_at_selected": float(sel.loc[best_epoch, "test_scc"]),
               "refit_mean_of_seeds": float(np.mean(per_seed)),
               "refit_sd_of_seeds": float(np.std(per_seed)),
               "refit_per_seed": [float(x) for x in per_seed],
               "refit_ensemble": float(
                   evaluate(alleles["test"], None, s["test"],
                            preds.mean(0))[0]["mean_allele_scc"]),
               "refit_ensemble_rank": float(evaluate(
                   alleles["test"], None, s["test"],
                   np.stack([within_allele_rank(q, alleles["test"]) for q in preds]).mean(0)
               )[0]["mean_allele_scc"]),
               "stage1_oracle_peak": float(sel["test_scc"].max()),
               "stage1_oracle_epoch": int(sel["test_scc"].idxmax())}
        summary.append(res)
        np.save(os.path.join(args.output_dir, f"preds_{model_name}_{target}.npy"), preds)
        print(f"{cfg:16s} val picks ep {best_epoch:3d} -> refit seeds "
              f"{res['refit_mean_of_seeds']:.4f} +/- {res['refit_sd_of_seeds']:.4f} | "
              f"ENSEMBLE {res['refit_ensemble']:.4f} | (stage1 test {res['stage1_test_at_selected']:.4f}, "
              f"oracle {res['stage1_oracle_peak']:.4f}@{res['stage1_oracle_epoch']})", flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "select_seeds": args.select_seeds,
                   "val_seed": args.val_seed, "max_epochs": args.epochs,
                   "peptide_only_null": null, "baseline_reference": 0.4156,
                   "configs": summary}, f, indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)
    np.save(os.path.join(args.output_dir, "test_alleles.npy"), alleles["test"])
    np.save(os.path.join(args.output_dir, "test_s.npy"), s["test"])
    print(f"\n[*] Wrote summary.json, curves.csv and predictions to {args.output_dir}/")


if __name__ == "__main__":
    main()
