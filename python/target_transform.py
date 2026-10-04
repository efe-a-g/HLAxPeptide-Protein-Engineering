#!/usr/bin/env python3
"""
Match the training target to the metric, and ensemble over seeds.

The headline metric is the mean *within-allele* Spearman on held-out alleles. It is invariant
to any monotone per-allele transform of the prediction, so the between-allele offset in the
target -- 37% of its variance -- is signal the model spends capacity on and the metric then
throws away. These targets strip progressively more of it:

  raw       s = 2^(-1/t)                                  baseline.py's target, sigmoid output
  centered  s - mean_a(s)                                 removes the per-allele offset
  zscore    (s - mean_a(s)) / std_a(s)                     also equalises per-allele spread
  rank      within-allele average rank -> N(0,1) quantile  removes every monotone per-allele
                                                           distortion; what Spearman measures

All per-allele statistics come from training rows only, so nothing about the held-out alleles
leaks. Each is a strictly increasing map within an allele and so cannot change within-allele
order; the model predicts a *relative* score, which is all the metric reads.

Orthogonally: every result in this repo so far is the mean of per-seed scores. This also
reports the score of the ensemble of those same seeds (mean prediction, and mean of
within-allele ranks), which costs no extra training.

Split, features (one-hot peptide + 34-aa pseudosequence), network, optimiser and epoch count
are baseline.py's, unchanged. Train and held-out SCC are logged during training.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata, norm
from torch.utils.data import DataLoader, TensorDataset

from baseline import (StabilityNet, encode_sequences, evaluate, peptide_only_null,
                      resolve_device, split_by_supertype, transform_target)
from ranking_loss import LinearOutNet

# Each target is a monotone per-allele map of s; `sigmoid` says whether the output head can
# represent its range (raw is in [0, 1], the rest are centred on 0 and take negative values).
TARGETS = {
    "raw":      dict(sigmoid=True),
    "centered": dict(sigmoid=False),
    "zscore":   dict(sigmoid=False),
    "rank":     dict(sigmoid=False),
}


def make_target(name, s, allele_ids):
    g = pd.Series(s).groupby(allele_ids)
    if name == "raw":
        return s.astype(np.float32)
    if name == "centered":
        return (s - g.transform("mean").to_numpy()).astype(np.float32)
    if name == "zscore":
        sd = g.transform("std").to_numpy().copy()
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        return ((s - g.transform("mean").to_numpy()) / sd).astype(np.float32)
    if name == "rank":
        # Average ranks so the exact zeros and the 0.1 h floor stay tied, then map to normal
        # quantiles: an allele's targets become N(0,1)-shaped regardless of its own scale.
        out = np.empty(len(s), dtype=np.float32)
        for _, idx in pd.Series(np.arange(len(s))).groupby(allele_ids):
            i = idx.to_numpy()
            out[i] = norm.ppf(rankdata(s[i]) / (len(i) + 1.0))
        return out
    raise ValueError(name)


def within_allele_rank(pred, alleles):
    """Average-rank of each prediction within its allele, scaled to [0, 1]."""
    out = np.empty(len(pred), dtype=np.float64)
    for _, idx in pd.Series(np.arange(len(pred))).groupby(alleles):
        i = idx.to_numpy()
        out[i] = rankdata(pred[i]) / (len(i) + 1.0)
    return out


def train(cfg, X_tr, y_tr, X_te, args, seed, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    model = (StabilityNet(X_tr.shape[1], args.hidden_dim) if cfg["sigmoid"]
             else LinearOutNet(X_tr.shape[1], args.hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=args.batch_size, shuffle=True)
    Xtr_t, Xte_t = torch.from_numpy(X_tr).to(device), torch.from_numpy(X_te).to(device)

    def infer(T):
        model.eval()
        with torch.no_grad():
            return model(T).cpu().numpy()

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
        sched.step(total / len(y_tr))
        if log is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            tr = evaluate(log["tr_alleles"], None, log["s_tr"], infer(Xtr_t))[0]["mean_allele_scc"]
            te = evaluate(log["te_alleles"], None, log["s_te"], infer(Xte_t))[0]["mean_allele_scc"]
            curve.append({"epoch": epoch, "train_scc": tr, "test_scc": te})
    return infer(Xte_t), curve


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/target_transform")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    X_tr = encode_sequences(train_df["peptide"].values, train_df["hla_pseudoseq"].values,
                            encoding="sparse")
    X_te = encode_sequences(test_df["peptide"].values, test_df["hla_pseudoseq"].values,
                            encoding="sparse")
    s_tr = transform_target(train_df["thalf_hours"].values)
    s_te = transform_target(test_df["thalf_hours"].values)
    allele_ids = pd.factorize(train_df["allele"])[0]
    tr_alleles, te_alleles = train_df["allele"].to_numpy(), test_df["allele"].to_numpy()
    te_thalf = test_df["thalf_hours"].to_numpy()
    log = {"tr_alleles": tr_alleles, "te_alleles": te_alleles, "s_tr": s_tr, "s_te": s_te}

    null = evaluate(te_alleles, te_thalf, s_te,
                    peptide_only_null(train_df, test_df, s_tr))[0]["mean_allele_scc"]
    print(f"[*] peptide-only null {null:.4f}\n")

    summary, curves, preds_out = [], [], {}
    for name in args.targets:
        cfg = TARGETS[name]
        y_tr = make_target(name, s_tr, allele_ids)
        preds, per_seed = [], []
        for seed in args.seeds:
            pred, curve = train(cfg, X_tr, y_tr, X_te, args, seed, log)
            preds.append(pred)
            curves += [{"target": name, "seed": seed, **r} for r in curve]
            per_seed.append(evaluate(te_alleles, te_thalf, s_te, pred)[0]["mean_allele_scc"])
        preds = np.stack(preds)
        preds_out[name] = preds

        ens_mean = evaluate(te_alleles, te_thalf, s_te, preds.mean(0))[0]["mean_allele_scc"]
        ens_rank = evaluate(te_alleles, te_thalf, s_te,
                            np.stack([within_allele_rank(p, te_alleles) for p in preds]).mean(0)
                            )[0]["mean_allele_scc"]
        res = {"target": name, "mean_of_seeds": float(np.mean(per_seed)),
               "sd_of_seeds": float(np.std(per_seed)), "per_seed": [float(x) for x in per_seed],
               "ensemble_mean_pred": float(ens_mean), "ensemble_mean_rank": float(ens_rank)}
        if curves:
            c = pd.DataFrame([r for r in curves if r["target"] == name])
            m = c.groupby("epoch")[["train_scc", "test_scc"]].mean()
            res |= {"final_train_scc": float(m["train_scc"].iloc[-1]),
                    "peak_test_scc": float(m["test_scc"].max()),
                    "peak_epoch": int(m["test_scc"].idxmax())}
            res["scc_gap"] = res["final_train_scc"] - res["mean_of_seeds"]
        summary.append(res)
        print(f"{name:9s} seeds {res['mean_of_seeds']:.4f} +/- {res['sd_of_seeds']:.4f} | "
              f"ensemble(mean) {ens_mean:.4f} | ensemble(rank) {ens_rank:.4f} | "
              f"train {res.get('final_train_scc', float('nan')):.3f} "
              f"gap {res.get('scc_gap', float('nan')):.3f} | "
              f"peak {res.get('peak_test_scc', float('nan')):.4f}@{res.get('peak_epoch', -1)}",
              flush=True)

    np.savez(os.path.join(args.output_dir, "test_predictions.npz"),
             alleles=te_alleles, s_te=s_te, **preds_out)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "peptide_only_null": null, "variants": summary}, f,
                  indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)
    print(f"\n[*] Wrote summary.json, curves.csv, test_predictions.npz to {args.output_dir}/")


if __name__ == "__main__":
    main()
