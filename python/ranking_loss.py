#!/usr/bin/env python3
"""
Does focusing the loss on within-allele ranking help? Same split, features, hidden layer
(60 sigmoid units), optimiser and metric as baseline.py; only the target / loss changes.

  A  baseline      sigmoid output, MSE on s                      (baseline.py's setup)
  B  linear MSE    linear output, MSE on s                       (control for dropping the sigmoid)
  C  centered MSE  linear output, MSE on s - mean_allele(s)      (removes the allele offset)
  D  centered+rank C plus a within-allele margin ranking loss
  E  sigmoid squeezed  sigmoid output, target 0.5 * (s - mean_allele(s)) + 0.5, which stays in [0, 1]

B, C and D draw each batch from a few alleles (`--alleles_per_batch`) so that a batch holds
many same-allele pairs for the ranking term; B and C use the same batching so D vs C isolates
the ranking term and C vs B isolates centering. A uses ordinary shuffled batches.

Centering uses training-allele means of s only, so nothing about the held-out alleles leaks
in. It is a per-allele constant shift and cannot change within-allele order; the model then
predicts *relative* scores only, which is all per-allele Spearman looks at.

Ranking pairs are same-allele rows whose targets differ by at least `--min_diff`: the exact
zeros and the 0.1 h floor are effectively ties and must not be forced apart.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import (StabilityNet, encode_sequences, evaluate, resolve_device,
                      split_by_supertype, transform_target)


class LinearOutNet(nn.Module):
    """The baseline's hidden layer with a linear (unbounded) output."""

    def __init__(self, in_features=860, hidden_dim=60):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


VARIANTS = {
    "A baseline":       dict(sigmoid=True,  grouped=False, center=False, rank=0.0),
    "B linear MSE":     dict(sigmoid=False, grouped=True,  center=False, rank=0.0),
    "C centered MSE":   dict(sigmoid=False, grouped=True,  center=True,  rank=0.0),
    "D centered+rank":  dict(sigmoid=False, grouped=True,  center=True,  rank=1.0),
    # Sigmoid output cannot fit negative targets, so squeeze the centered target into [0, 1]:
    # 0.5 * (s - mean_allele(s)) + 0.5. Still a per-allele monotone map.
    "E sigmoid squeezed": dict(sigmoid=True, grouped=True,  center="squeeze", rank=0.0),
}


def pairwise_rank_loss(z, s, a, margin, min_diff):
    """Margin ranking loss over same-allele pairs where row i is clearly better than row j."""
    ds = s[:, None] - s[None, :]
    mask = (a[:, None] == a[None, :]) & (ds >= min_diff)
    i, j = mask.nonzero(as_tuple=True)
    if len(i) == 0:
        return z.sum() * 0.0
    return F.margin_ranking_loss(z[i], z[j], torch.ones(len(i), device=z.device), margin=margin)


def grouped_batch(rng, rows_by_allele, probs, n_alleles, k, per):
    picks = rng.choice(n_alleles, size=k, replace=False, p=probs)
    return np.concatenate([rng.choice(rows_by_allele[a], size=min(per, len(rows_by_allele[a])),
                                      replace=False) for a in picks])


def train_variant(cfg, X, target, s, allele_ids, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)
    model = (StabilityNet(X.shape[1], args.hidden_dim) if cfg["sigmoid"]
             else LinearOutNet(X.shape[1], args.hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(target).to(device)
    st, at = torch.from_numpy(s).to(device), torch.from_numpy(allele_ids).to(device)
    n = len(X)
    n_alleles = int(allele_ids.max()) + 1
    rows_by_allele = [np.flatnonzero(allele_ids == a) for a in range(n_alleles)]
    probs = np.array([len(r) for r in rows_by_allele], dtype=float)
    probs /= probs.sum()                      # alleles drawn in proportion to their row counts
    per = args.batch_size // args.alleles_per_batch
    n_steps = math.ceil(n / args.batch_size)

    for _ in range(args.epochs):
        model.train()
        if cfg["grouped"]:
            batches = [grouped_batch(rng, rows_by_allele, probs, n_alleles,
                                     args.alleles_per_batch, per) for _ in range(n_steps)]
        else:
            perm = rng.permutation(n)
            batches = [perm[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
        total = 0.0
        for idx in batches:
            idx = torch.from_numpy(idx).to(device)
            out = model(Xt[idx])
            loss = F.mse_loss(out, yt[idx])
            if cfg["rank"] > 0:
                loss = loss + cfg["rank"] * pairwise_rank_loss(
                    out, st[idx], at[idx], args.margin, args.min_diff)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step(total / (n_steps * args.batch_size))

    def predict(Xe):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(Xe).to(device)).cpu().numpy()
    return predict


def between_share(values, groups):
    """Share of the variance of `values` that sits between groups."""
    s = pd.Series(values)
    return float(((s.groupby(groups).transform("mean") - s.mean()) ** 2).sum()
                 / ((s - s.mean()) ** 2).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--margin", type=float, default=0.05,
                   help="Margin in output units (targets are s in [0, 1])")
    p.add_argument("--min_diff", type=float, default=0.02,
                   help="Pairs closer than this in s are treated as ties and skipped")
    p.add_argument("--alleles_per_batch", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/ranking_loss")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    X_tr = encode_sequences(train_df["peptide"].values, train_df["hla_pseudoseq"].values)
    X_te = encode_sequences(test_df["peptide"].values, test_df["hla_pseudoseq"].values)
    s_tr = transform_target(train_df["thalf_hours"].values)
    y_te = transform_target(test_df["thalf_hours"].values)
    allele_ids = pd.factorize(train_df["allele"])[0].astype(np.int64)
    centered = (s_tr - pd.Series(s_tr).groupby(allele_ids).transform("mean").to_numpy()
                ).astype(np.float32)
    test_alleles, test_thalf = test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy()
    print(f"[*] allele-mean offset is {between_share(s_tr, allele_ids):.1%} of target variance "
          f"in train; removed by centering")

    summary, per_allele = [], []
    for name in args.variants:
        cfg = VARIANTS[name]
        target = {False: s_tr, True: centered,
                  "squeeze": (0.5 * centered + 0.5).astype(np.float32)}[cfg["center"]]
        scores, shares, per = [], [], []
        for seed in args.seeds:
            predict = train_variant(cfg, X_tr, target, s_tr, allele_ids, args, seed)
            pred = predict(X_te)
            m, pa = evaluate(test_alleles, test_thalf, y_te, pred)
            scores.append(m["mean_allele_scc"])
            shares.append(between_share(pred, test_alleles))
            per.append(pa.set_index("allele")["scc"])
        per_allele.append(pd.concat(per, axis=1).mean(axis=1).rename(name))
        res = {"variant": name, "scc_mean": float(np.mean(scores)),
               "scc_sd": float(np.std(scores)), "scc_per_seed": [float(x) for x in scores],
               "pred_between_allele_share": float(np.mean(shares))}
        summary.append(res)
        print(f"{name:16s} SCC {res['scc_mean']:.3f} +/- {res['scc_sd']:.3f}  "
              f"(seeds: {', '.join(f'{x:.3f}' for x in scores)}) | "
              f"between-allele share of prediction variance {res['pred_between_allele_share']:.2f}",
              flush=True)

    truth_share = between_share(y_te, test_alleles)
    print(f"truth: between-allele share of target variance on test = {truth_share:.2f}")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "margin": args.margin, "min_diff": args.min_diff,
                   "alleles_per_batch": args.alleles_per_batch,
                   "truth_between_allele_share": truth_share, "variants": summary}, f, indent=2)
    pd.concat(per_allele, axis=1).to_csv(os.path.join(args.output_dir, "allele_scc.csv"))
    print(f"[*] Wrote summary.json and allele_scc.csv to {args.output_dir}/")


if __name__ == "__main__":
    main()
