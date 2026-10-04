#!/usr/bin/env python3
"""
Row 3 of the inverse-folding table: does the structural channel add to the trained baseline?

Three feature blocks come out of the mode-(b) matrices (mpnn_score.py), all free once the
matrices exist:

  logp9     log p(x_i | backbone, hla_seq) for the peptide's own residue at each of the 9
            positions. Per ROW, since it depends on the peptide. Expected to beat the scalar,
            because a sum dilutes P2/P9 across seven positions that barely matter.
  mean      the sum of those nine, one scalar per row.
  anchor    the full 20-way distributions at P2 and P-Omega, 40 values. Per ALLELE, constant
            within an allele: the model's statement of what the pocket wants, which is what
            the pseudosequence is a proxy for, arrived at from geometry instead.

Arms (--arms), all on the canonical split and the convergence protocol from
encoding_convergence.py (val alleles pick the epoch, refit on full train, read test once):

  baseline            one-hot peptide + pseudosequence             <- row 2
  mpnn_only           structural features alone, trained
  baseline+logp9 / +mean / +anchor / +all                          <- row 3

Row 3 is only interesting relative to row 2. If it adds nothing, the information was already
in the pseudosequence -- a clean result with a mechanism, not a null. Structural blocks are
standardised on train rows only; without that they are swamped by the 860-dim one-hot.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from baseline import evaluate, split_by_supertype, transform_target
from encoding_convergence import encode, run

AA20 = list("ARNDCQEGHILKMFPSTWYV")
IDX = {a: i for i, a in enumerate(AA20)}


def mpnn_blocks(df, mats):
    """-> dict of named feature blocks, each (len(df), d)."""
    pep = np.array([[IDX[c] for c in p] for p in df["peptide"]])
    M = np.stack([mats[a] for a in df["allele"]])              # (N, 9, 20)
    logp9 = np.take_along_axis(M, pep[:, :, None], 2).squeeze(2)   # (N, 9)
    anchor = np.concatenate([M[:, 1, :], M[:, 8, :]], 1)           # (N, 40) P2 and P9
    return {"logp9": logp9.astype(np.float32),
            "mean": logp9.sum(1, keepdims=True).astype(np.float32),
            "anchor": anchor.astype(np.float32)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--pssm", default="outputs/mpnn/pssm_by_allele.npz")
    p.add_argument("--arms", nargs="+",
                   default=["baseline", "mpnn_only", "baseline+logp9", "baseline+mean",
                            "baseline+anchor", "baseline+all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--select_seeds", type=int, nargs="+", default=[42])
    p.add_argument("--val_seed", type=int, default=7)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="outputs/mpnn_integrate")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    z = np.load(args.pssm, allow_pickle=True)
    mats = dict(zip(z["allele"], z["pssm"]))
    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df = df.iloc[train_pos].reset_index(drop=True)
    test_df = df.iloc[test_pos]
    print("[*] inner fit/val split within train:")
    fit_pos, val_pos = split_by_supertype(train_df, seed=args.val_seed)
    frames = {"fit": train_df.iloc[fit_pos], "val": train_df.iloc[val_pos],
              "train": train_df, "test": test_df}
    s = {k: transform_target(d["thalf_hours"].values) for k, d in frames.items()}
    alleles = {k: d["allele"].to_numpy() for k, d in frames.items()}

    base = {k: encode(d, "onehot") for k, d in frames.items()}
    blocks = {k: mpnn_blocks(d, mats) for k, d in frames.items()}
    # Standardise each structural block on the rows the model actually trains on.
    stats = {}
    for name in ("logp9", "mean", "anchor"):
        a = blocks["train"][name]
        mu, sd = a.mean(0), a.std(0)
        sd[sd < 1e-8] = 1.0
        stats[name] = (mu, sd)

    def build(split, arm):
        parts = []
        if arm != "mpnn_only":
            parts.append(base[split])
        names = (["logp9", "mean", "anchor"] if arm in ("mpnn_only", "baseline+all")
                 else ([arm.split("+")[1]] if "+" in arm else []))
        for n in names:
            mu, sd = stats[n]
            parts.append((blocks[split][n] - mu) / sd)
        return np.hstack(parts).astype(np.float32)

    print()
    summary = []
    for arm in args.arms:
        X = {k: build(k, arm) for k in frames}
        sets1 = {k: {"X": X[k], "alleles": alleles[k], "s": s[k]} for k in ("fit", "val", "test")}
        sets2 = {k: {"X": X[k], "alleles": alleles[k], "s": s[k]} for k in ("train", "test")}
        sel = []
        for seed in args.select_seeds:
            c, _ = run(X["fit"], s["fit"], sets1, args, seed)
            sel.append(c)
        sel = sum(sel) / len(sel)
        best = int(sel["val_scc"].idxmax())

        ep_backup, args.epochs = args.epochs, best
        preds, per_seed, tr = [], [], []
        for seed in args.seeds:
            c, stored = run(X["train"], s["train"], sets2, args, seed)
            preds.append(stored[best])
            per_seed.append(evaluate(alleles["test"], None, s["test"],
                                     stored[best])[0]["mean_allele_scc"])
            tr.append(c.loc[best, "train_scc"])
        args.epochs = ep_backup
        preds = np.stack(preds)
        res = {"arm": arm, "input_dim": int(X["train"].shape[1]), "selected_epoch": best,
               "mean_of_seeds": float(np.mean(per_seed)), "sd_of_seeds": float(np.std(per_seed)),
               "ensemble": float(evaluate(alleles["test"], None, s["test"],
                                          preds.mean(0))[0]["mean_allele_scc"]),
               "train_scc": float(np.mean(tr))}
        res["gap"] = res["train_scc"] - res["mean_of_seeds"]
        summary.append(res)
        np.save(os.path.join(args.output_dir, f"preds_{arm.replace('+','_')}.npy"), preds)
        print(f"{arm:18s} dim {res['input_dim']:5d} ep {best:3d} | test {res['mean_of_seeds']:.4f} "
              f"+/- {res['sd_of_seeds']:.4f} | ENS {res['ensemble']:.4f} | "
              f"train {res['train_scc']:.3f} | gap {res['gap']:.3f}", flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "arms": summary}, f, indent=2)
    np.save(os.path.join(args.output_dir, "test_alleles.npy"), alleles["test"])
    np.save(os.path.join(args.output_dir, "test_s.npy"), s["test"])
    print(f"\n[*] wrote to {args.output_dir}/")


if __name__ == "__main__":
    main()
