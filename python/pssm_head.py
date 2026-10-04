#!/usr/bin/env python3
"""
Architectures that give the model the structure of the problem instead of a flat vector.

baseline.py concatenates peptide and HLA one-hots and hands them to a dense MLP, which has to
discover from 58 training alleles both that the two halves are different kinds of thing and
that they interact. The classical description of class I binding is a position-specific
scoring matrix per allele: the HLA defines a motif, the peptide is scored against it. These
variants build that in:

  mlp       baseline.py's dense net on the concatenation                       (reference)
  pssm      HLA -> 9x20 scoring matrix; score = sum of the peptide's entries.  Additive over
            peptide positions, and every parameter is allele-conditioned, so the model cannot
            score a peptide without using the allele.
  bilinear  peptide tower . HLA tower (rank-k dot product)
  pssm_mlp  pssm plus a small dense correction, for non-additive position effects

A PSSM is linear in the peptide one-hot *given* the allele, so the whole model is a hypernet:
all its peptide-side capacity is spent on how the motif changes with the allele, which is the
only thing that transfers to a held-out allele.

Split, features, optimiser and epoch count are baseline.py's. Target defaults to within-allele
centering (--target, see target_transform.py), since the metric is within-allele ranking.
Reports per-seed scores, the seed-ensemble score, and the train/held-out SCC gap.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (AA_TO_IDX, evaluate, peptide_only_null, resolve_device,
                      split_by_supertype, transform_target)
from target_transform import make_target, within_allele_rank

N_POS, N_AA, HLA_LEN = 9, 20, 34


def encode_parts(df):
    """Peptide as integer indices (for gathering), HLA pseudosequence as a flat one-hot."""
    pep = np.array([[AA_TO_IDX[a] for a in p] for p in df["peptide"]], dtype=np.int64)
    hla_idx = np.array([[AA_TO_IDX[a] for a in h] for h in df["hla_pseudoseq"]], dtype=np.int64)
    hla = (np.eye(N_AA, dtype=np.float32)[hla_idx] * 0.85 + 0.05).reshape(len(df), -1)
    pep_oh = (np.eye(N_AA, dtype=np.float32)[pep] * 0.85 + 0.05).reshape(len(df), -1)
    return pep, pep_oh, hla


class MLP(nn.Module):
    """baseline.py's network, linear output (the target is centred, so it takes both signs)."""

    def __init__(self, pep_dim, hla_dim, hidden=60, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(pep_dim + hla_dim, hidden), nn.Sigmoid(),
                                 nn.Linear(hidden, 1))

    def forward(self, pep_idx, pep_oh, hla):
        return self.net(torch.cat([pep_oh, hla], 1)).squeeze(-1)


class PSSM(nn.Module):
    """HLA -> (9, 20) scoring matrix; the peptide's score is the sum of its own entries."""

    def __init__(self, pep_dim, hla_dim, hidden=60, **_):
        super().__init__()
        self.hla_net = nn.Sequential(nn.Linear(hla_dim, hidden), nn.Sigmoid(),
                                     nn.Linear(hidden, N_POS * N_AA))

    def score_matrix(self, hla):
        return self.hla_net(hla).view(-1, N_POS, N_AA)

    def forward(self, pep_idx, pep_oh, hla):
        pssm = self.score_matrix(hla)
        return pssm.gather(2, pep_idx.unsqueeze(-1)).squeeze(-1).sum(1)


class Bilinear(nn.Module):
    """Rank-k dot product between a peptide embedding and an HLA embedding."""

    def __init__(self, pep_dim, hla_dim, hidden=60, rank=32, **_):
        super().__init__()
        self.pep = nn.Sequential(nn.Linear(pep_dim, hidden), nn.Sigmoid(), nn.Linear(hidden, rank))
        self.hla = nn.Sequential(nn.Linear(hla_dim, hidden), nn.Sigmoid(), nn.Linear(hidden, rank))

    def forward(self, pep_idx, pep_oh, hla):
        return (self.pep(pep_oh) * self.hla(hla)).sum(1)


class PSSMPlusMLP(nn.Module):
    """PSSM score plus a small dense correction on the concatenation."""

    def __init__(self, pep_dim, hla_dim, hidden=60, corr_hidden=32, **_):
        super().__init__()
        self.pssm = PSSM(pep_dim, hla_dim, hidden)
        self.corr = nn.Sequential(nn.Linear(pep_dim + hla_dim, corr_hidden), nn.Sigmoid(),
                                  nn.Linear(corr_hidden, 1))

    def forward(self, pep_idx, pep_oh, hla):
        return self.pssm(pep_idx, pep_oh, hla) + self.corr(torch.cat([pep_oh, hla], 1)).squeeze(-1)


MODELS = {"mlp": MLP, "pssm": PSSM, "bilinear": Bilinear, "pssm_mlp": PSSMPlusMLP}


def train(make, parts_tr, y_tr, parts_te, args, seed, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    pep_tr, pep_oh_tr, hla_tr = (torch.from_numpy(a).to(device) for a in parts_tr)
    pep_te, pep_oh_te, hla_te = (torch.from_numpy(a).to(device) for a in parts_te)
    model = make(pep_oh_tr.shape[1], hla_tr.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()
    yt = torch.from_numpy(y_tr).to(device)
    loader = DataLoader(TensorDataset(torch.arange(len(y_tr))), batch_size=args.batch_size,
                        shuffle=True)

    def infer(parts):
        model.eval()
        with torch.no_grad():
            return model(*parts).cpu().numpy()

    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (idx,) in loader:
            idx = idx.to(device)
            opt.zero_grad()
            loss = crit(model(pep_tr[idx], pep_oh_tr[idx], hla_tr[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step(total / len(y_tr))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            tr = evaluate(log["tr_alleles"], None, log["s_tr"],
                          infer((pep_tr, pep_oh_tr, hla_tr)))[0]["mean_allele_scc"]
            te = evaluate(log["te_alleles"], None, log["s_te"],
                          infer((pep_te, pep_oh_te, hla_te)))[0]["mean_allele_scc"]
            curve.append({"epoch": epoch, "train_scc": tr, "test_scc": te})
    n_params = sum(p.numel() for p in model.parameters())
    return infer((pep_te, pep_oh_te, hla_te)), curve, n_params


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--target", default="centered", help="see target_transform.TARGETS")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/pssm_head")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    parts_tr, parts_te = encode_parts(train_df), encode_parts(test_df)
    s_tr = transform_target(train_df["thalf_hours"].values)
    s_te = transform_target(test_df["thalf_hours"].values)
    y_tr = make_target(args.target, s_tr, pd.factorize(train_df["allele"])[0])
    tr_alleles, te_alleles = train_df["allele"].to_numpy(), test_df["allele"].to_numpy()
    te_thalf = test_df["thalf_hours"].to_numpy()
    log = {"tr_alleles": tr_alleles, "te_alleles": te_alleles, "s_tr": s_tr, "s_te": s_te}

    null = evaluate(te_alleles, te_thalf, s_te,
                    peptide_only_null(train_df, test_df, s_tr))[0]["mean_allele_scc"]
    print(f"[*] target={args.target} | peptide-only null {null:.4f}\n")

    summary, curves, preds_out = [], [], {}
    for name in args.models:
        preds, per_seed, n_params = [], [], 0
        for seed in args.seeds:
            pred, curve, n_params = train(MODELS[name], parts_tr, y_tr, parts_te, args, seed, log)
            preds.append(pred)
            curves += [{"model": name, "seed": seed, **r} for r in curve]
            per_seed.append(evaluate(te_alleles, te_thalf, s_te, pred)[0]["mean_allele_scc"])
        preds = np.stack(preds)
        preds_out[name] = preds
        m = pd.DataFrame([r for r in curves if r["model"] == name]
                         ).groupby("epoch")[["train_scc", "test_scc"]].mean()
        res = {"model": name, "n_params": int(n_params),
               "mean_of_seeds": float(np.mean(per_seed)), "sd_of_seeds": float(np.std(per_seed)),
               "per_seed": [float(x) for x in per_seed],
               "ensemble_mean_pred": float(
                   evaluate(te_alleles, te_thalf, s_te, preds.mean(0))[0]["mean_allele_scc"]),
               "ensemble_mean_rank": float(evaluate(
                   te_alleles, te_thalf, s_te,
                   np.stack([within_allele_rank(q, te_alleles) for q in preds]).mean(0)
               )[0]["mean_allele_scc"]),
               "final_train_scc": float(m["train_scc"].iloc[-1]),
               "peak_test_scc": float(m["test_scc"].max()), "peak_epoch": int(m["test_scc"].idxmax())}
        res["scc_gap"] = res["final_train_scc"] - res["mean_of_seeds"]
        summary.append(res)
        print(f"{name:9s} params {res['n_params']:6d} | seeds {res['mean_of_seeds']:.4f} +/- "
              f"{res['sd_of_seeds']:.4f} | ens(mean) {res['ensemble_mean_pred']:.4f} | "
              f"ens(rank) {res['ensemble_mean_rank']:.4f} | train {res['final_train_scc']:.3f} "
              f"gap {res['scc_gap']:.3f} | peak {res['peak_test_scc']:.4f}@{res['peak_epoch']}",
              flush=True)

    np.savez(os.path.join(args.output_dir, "test_predictions.npz"),
             alleles=te_alleles, s_te=s_te, **preds_out)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "target": args.target, "peptide_only_null": null,
                   "variants": summary}, f, indent=2)
    pd.DataFrame(curves).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)
    print(f"\n[*] Wrote summary.json, curves.csv, test_predictions.npz to {args.output_dir}/")


if __name__ == "__main__":
    main()
