#!/usr/bin/env python3
"""
Overfitting diagnostic for the loss variants in ranking_loss.py (A-E).

Re-runs each variant with the same split, features, batching, losses and optimiser (the
building blocks are imported from ranking_loss.py), but logs per-allele Spearman on the
*training* alleles and the held-out alleles every `--eval_every` epochs. Losses differ across
variants (different targets), so MSE is not comparable between them; the train-vs-held-out
SCC gap and the shape of the held-out curve are.

Held-out curves are for diagnosis only; the reported score is the final epoch.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from baseline import (StabilityNet, encode_sequences, evaluate, resolve_device,
                      split_by_supertype, transform_target)
from ranking_loss import VARIANTS, LinearOutNet, grouped_batch, pairwise_rank_loss


def run(cfg, X, target, s, allele_ids, X_te, ev, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)
    model = (StabilityNet(X.shape[1], args.hidden_dim) if cfg["sigmoid"]
             else LinearOutNet(X.shape[1], args.hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(target).to(device)
    st, at = torch.from_numpy(s).to(device), torch.from_numpy(allele_ids).to(device)
    Xe = torch.from_numpy(X_te).to(device)
    n, n_alleles = len(X), int(allele_ids.max()) + 1
    rows_by_allele = [np.flatnonzero(allele_ids == a) for a in range(n_alleles)]
    probs = np.array([len(r) for r in rows_by_allele], dtype=float)
    probs /= probs.sum()
    per = args.batch_size // args.alleles_per_batch
    n_steps = math.ceil(n / args.batch_size)

    def predict(Xs):
        model.eval()
        with torch.no_grad():
            return model(Xs).cpu().numpy()

    curve = []
    for epoch in range(1, args.epochs + 1):
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

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            tr, _ = evaluate(ev["train_alleles"], None, ev["train_s"], predict(Xt))
            te, _ = evaluate(ev["test_alleles"], ev["test_thalf"], ev["test_s"], predict(Xe))
            curve.append({"epoch": epoch, "train_scc": tr["mean_allele_scc"],
                          "test_scc": te["mean_allele_scc"]})
    return pd.DataFrame(curve).set_index("epoch")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--min_diff", type=float, default=0.02)
    p.add_argument("--alleles_per_batch", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="outputs/ranking_overfit")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    X_tr = encode_sequences(train_df["peptide"].values, train_df["hla_pseudoseq"].values)
    X_te = encode_sequences(test_df["peptide"].values, test_df["hla_pseudoseq"].values)
    s_tr = transform_target(train_df["thalf_hours"].values)
    allele_ids = pd.factorize(train_df["allele"])[0].astype(np.int64)
    centered = (s_tr - pd.Series(s_tr).groupby(allele_ids).transform("mean").to_numpy()
                ).astype(np.float32)
    ev = {"train_alleles": train_df["allele"].to_numpy(), "train_s": s_tr,
          "test_alleles": test_df["allele"].to_numpy(),
          "test_thalf": test_df["thalf_hours"].to_numpy(),
          "test_s": transform_target(test_df["thalf_hours"].values)}

    summary, rows = [], []
    for name in args.variants:
        cfg = VARIANTS[name]
        target = {False: s_tr, True: centered,
                  "squeeze": (0.5 * centered + 0.5).astype(np.float32)}[cfg["center"]]
        runs = []
        for seed in args.seeds:
            c = run(cfg, X_tr, target, s_tr, allele_ids, X_te, ev, args, seed)
            runs.append(c)
            rows += [{"variant": name, "seed": seed, "epoch": e, **r}
                     for e, r in c.iterrows()]
        mean = sum(runs) / len(runs)
        final = [r.iloc[-1] for r in runs]
        res = {"variant": name,
               "final_test_scc": float(np.mean([f["test_scc"] for f in final])),
               "final_test_scc_sd": float(np.std([f["test_scc"] for f in final])),
               "final_train_scc": float(np.mean([f["train_scc"] for f in final])),
               "peak_test_scc": float(mean["test_scc"].max()),
               "peak_epoch": int(mean["test_scc"].idxmax())}
        res["gap"] = res["final_train_scc"] - res["final_test_scc"]
        summary.append(res)
        print(f"{name:20s} test SCC {res['final_test_scc']:.3f} +/- {res['final_test_scc_sd']:.3f}"
              f" | train SCC {res['final_train_scc']:.3f} | gap {res['gap']:.3f} | "
              f"peak test {res['peak_test_scc']:.3f} @ epoch {res['peak_epoch']}", flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "variants": summary}, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "curves.csv"), index=False)


if __name__ == "__main__":
    main()
