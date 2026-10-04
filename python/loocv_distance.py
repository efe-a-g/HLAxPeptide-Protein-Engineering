#!/usr/bin/env python3
"""
Leave-one-allele-out: per-allele Spearman against how far that allele sits from the training set.

The headline split holds out 16 alleles once. This runs all 75 as their own fold, which gives
75 points instead of 16 and makes the distance relationship visible. For each held-out row the
distance is the reference protocol's: Hamming(182-aa hla_seq) + Hamming(9-mer peptide) to the
closest training row, averaged over the allele's rows.

Three models per fold:
  nn        no training. Predict the mean s of every training row at the minimum distance.
  baseline  baseline.py exactly -- one-hot peptide + 34-aa pseudosequence -> StabilityNet.
  combined  baseline's inputs plus four standardised neighbour features.

The neighbour features are [nn_pred, nn_dist, nn_hla_dist, nn_pep_dist]. For a TRAINING row
they are computed from the nearest row of a *different allele*, excluding the held-out allele.
Both exclusions matter: without the first, a training row's nearest neighbour is itself and
nn_pred leaks its own target; without the second, held-out targets leak in as feature values.
Test rows get the same rule for free, since every other-allele row is in the training set.

Distances factor, which is what makes 75 folds cheap: the HLA part depends only on the allele
pair, and the peptide part only on the peptide pair, so both are tabulated once up front.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baseline import AA_TO_IDX, StabilityNet, evaluate, transform_target

NN_FEATS = ["nn_pred", "nn_dist", "nn_hla", "nn_pep"]


def hamming_table(seqs):
    """(n, n) int16 Hamming distances between equal-length sequences."""
    idx = np.array([[AA_TO_IDX[a] for a in s] for s in seqs], dtype=np.int8)
    n, L = idx.shape
    out = np.empty((n, n), dtype=np.int16)
    for i in range(0, n, 512):
        out[i:i + 512] = (idx[i:i + 512, None, :] != idx[None, :, :]).sum(2)
    return out


def neighbour_tables(df, s, pep_d, hla_d, allele_ids, pep_ids, n_alleles):
    """For each (unique peptide, allele b): nearest peptide distance, and the tied rows' s.

    PEPMIN[p, b] is the smallest peptide Hamming from p to any row of allele b; CNT counts the
    rows achieving it and SUMS totals their s, so a tie-weighted mean survives the later
    min-over-alleles without revisiting rows.
    """
    n_pep = pep_d.shape[0]
    PEPMIN = np.zeros((n_pep, n_alleles), dtype=np.int16)
    SUMS = np.zeros((n_pep, n_alleles), dtype=np.float64)
    CNT = np.zeros((n_pep, n_alleles), dtype=np.int32)
    for b in range(n_alleles):
        rows = np.flatnonzero(allele_ids == b)
        sub = pep_d[:, pep_ids[rows]]
        m = sub.min(1)
        mask = sub == m[:, None]
        PEPMIN[:, b] = m
        CNT[:, b] = mask.sum(1)
        SUMS[:, b] = mask @ s[rows]
    return PEPMIN, SUMS, CNT


def nn_features(rows, held_out, allele_ids, pep_ids, hla_d, PEPMIN, SUMS, CNT):
    """Neighbour features for `rows`, using only alleles other than each row's own and held_out."""
    a = allele_ids[rows]
    p = pep_ids[rows]
    d = hla_d[a] + PEPMIN[p]                      # (len(rows), n_alleles)
    blocked = (a[:, None] == np.arange(d.shape[1])[None, :])
    if held_out is not None:
        blocked |= (np.arange(d.shape[1])[None, :] == held_out)
    d = np.where(blocked, np.int32(1 << 20), d.astype(np.int32))
    best = d.min(1)
    tie = d == best[:, None]
    cnt = np.where(tie, CNT[p], 0)
    tot = cnt.sum(1)
    pred = (np.where(tie, SUMS[p], 0.0).sum(1) / tot)
    hla = (np.where(tie, hla_d[a], 0) * cnt).sum(1) / tot
    out = np.stack([pred, best, hla, best - hla], 1).astype(np.float32)
    return out


def train_net(X_tr, y_tr, X_te, epochs, seed, hidden=60, lr=1e-3, batch=128):
    torch.set_num_threads(1)   # workers are run one per core; threading here only contends
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    model = StabilityNet(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()
    n = len(yt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step(total / n)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_te)).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--folds", default="all", help="'all' or e.g. 0-24")
    p.add_argument("--epochs", type=int, default=420)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/loocv_distance")
    p.add_argument("--tag", default="all")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    s = transform_target(df["thalf_hours"].values)
    alleles = np.sort(df["allele"].unique())
    allele_ids = pd.Categorical(df["allele"], categories=alleles).codes.astype(np.int32)
    peps = np.sort(df["peptide"].unique())
    pep_ids = pd.Categorical(df["peptide"], categories=peps).codes.astype(np.int32)

    one = df.drop_duplicates("allele").set_index("allele").loc[alleles]
    hla_d = hamming_table(one["hla_seq"].to_numpy())
    pep_d = hamming_table(peps).astype(np.int16)
    PEPMIN, SUMS, CNT = neighbour_tables(df, s, pep_d, hla_d, allele_ids, pep_ids, len(alleles))
    del pep_d

    idx = np.array([[AA_TO_IDX[a] for a in pp + hh]
                    for pp, hh in zip(df["peptide"], df["hla_pseudoseq"])])
    X_base = (np.eye(20, dtype=np.float32)[idx] * 0.85 + 0.05).reshape(len(df), -1)

    folds = (range(len(alleles)) if args.folds == "all"
             else range(*(int(x) for x in args.folds.split("-"))))
    rows_out = []
    preds_out = {}
    for a in folds:
        te = np.flatnonzero(allele_ids == a)
        tr = np.flatnonzero(allele_ids != a)
        f_te = nn_features(te, a, allele_ids, pep_ids, hla_d, PEPMIN, SUMS, CNT)
        f_tr = nn_features(tr, a, allele_ids, pep_ids, hla_d, PEPMIN, SUMS, CNT)
        mu, sd = f_tr.mean(0), f_tr.std(0)
        sd[sd < 1e-8] = 1.0
        Xc_tr = np.hstack([X_base[tr], (f_tr - mu) / sd]).astype(np.float32)
        Xc_te = np.hstack([X_base[te], (f_te - mu) / sd]).astype(np.float32)

        pred = {"nn": f_te[:, 0].astype(np.float64),
                "baseline": train_net(X_base[tr], s[tr], X_base[te], args.epochs, args.seed),
                "combined": train_net(Xc_tr, s[tr], Xc_te, args.epochs, args.seed)}
        al_te = df["allele"].to_numpy()[te]
        row = {"allele": alleles[a], "n": len(te),
               "mean_hamming": float(f_te[:, 1].mean()),
               "mean_hla_hamming": float(f_te[:, 2].mean()),
               "mean_peptide_hamming": float(f_te[:, 3].mean())}
        for k, v in pred.items():
            row[f"{k}_spearman"] = evaluate(al_te, None, s[te], v)[0]["mean_allele_scc"]
            preds_out[f"{alleles[a]}|{k}"] = v
        rows_out.append(row)
        print(f"{alleles[a]:22s} n {len(te):5d} dist {row['mean_hamming']:6.2f} | "
              f"nn {row['nn_spearman']:+.3f} base {row['baseline_spearman']:+.3f} "
              f"comb {row['combined_spearman']:+.3f}", flush=True)

    out = pd.DataFrame(rows_out)
    out.to_csv(os.path.join(args.output_dir, f"per_allele_{args.tag}.csv"), index=False)
    np.savez(os.path.join(args.output_dir, f"preds_{args.tag}.npz"),
             s=s, allele=df["allele"].to_numpy(), **preds_out)
    with open(os.path.join(args.output_dir, f"protocol_{args.tag}.json"), "w") as f:
        json.dump({"epochs": args.epochs, "seed": args.seed, "folds": str(args.folds),
                   "distance": "Hamming(182-aa hla_seq) + Hamming(9-mer peptide)",
                   "train_feature_rule": "nearest row of a different allele, held-out excluded",
                   "metric": "within-allele Spearman, all rows"}, f, indent=2)
    print(f"\n[*] wrote {args.output_dir}/per_allele_{args.tag}.csv")


if __name__ == "__main__":
    main()
