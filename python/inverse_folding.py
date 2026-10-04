#!/usr/bin/env python3
"""
Does adding inverse-folding (LigandMPNN) features to the baseline network help?

The features were computed once on the `inverse-folding-slides` branch (5 PDB templates x 3
decoding orders, alpha1/alpha2 only; see data/inverse_folding/provenance.json) and cover every row
of the dataset. Per row:

  scalar        1   mean log-probability of the observed peptide
  per_position  9   log-probability at each peptide position
  anchor       40   full 20-way softmax at P2 and P9
  all          50   the three together

Two scoring modes: `uncond` (peptide identities hidden, HLA visible: p(x_i | backbone, HLA)) and
`cond` (autoregressive, so a position sees earlier peptide residues). `scrambled` is the control
where the HLA sequence was shuffled before scoring.

Everything else is baseline.py: same split (split_by_supertype), BLOSUM50 peptide + pseudosequence
input, 60 sigmoid hidden units, MSE on s = 2^(-t0/t_half), Adam 1e-3, weight decay 1e-5, batch 128,
ReduceLROnPlateau on the running train loss. The 50 features are z-scored with *training-row*
statistics and appended to the 860 inputs.

Convergence. The baseline's 90 epochs is not converged (train MSE is still falling). Here each run
trains until the learning rate has been halved down to `--min_lr` (1e-6) or `--max_epochs`, a rule
that only sees the training loss, so the held-out alleles never choose when to stop. The held-out
score at epoch 90 is recorded from the same run for comparison with the 90-epoch protocol.

  python python/inverse_folding.py --configs baseline uncond_all cond_all scrambled_all
  python python/inverse_folding.py --report        # aggregate everything under the output dir
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (StabilityNet, encode_sequences, evaluate, resolve_device,
                      split_by_supertype, transform_target)
from overfitting_curves import snapshot

FEATURE_SETS = ("scalar", "per_position", "anchor", "all")
MODES = {"uncond": "features_unconditional.npz", "cond": "features_conditional.npz",
         "scrambled": "features_scrambled.npz"}
ALL_CONFIGS = (["baseline"] + [f"{m}_{fs}" for m in ("uncond", "cond") for fs in FEATURE_SETS]
               + ["scrambled_all"])


def load_features(features_dir, mode, feature_set):
    z = np.load(os.path.join(features_dir, MODES[mode]))
    parts = {"scalar": z["mean_logp"][:, None], "per_position": z["per_position"],
             "anchor": z["anchor_softmax"]}
    names = ("scalar", "per_position", "anchor") if feature_set == "all" else (feature_set,)
    return np.hstack([parts[n] for n in names]).astype(np.float32)


def train_converged(X_tr, y_tr, X_te, y_te, train_df, test_df, seed, args):
    """Baseline training, run until the LR schedule has decayed; returns (model, curve rows)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    model = StabilityNet(X_tr.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=args.batch_size, shuffle=True)
    Xtr_t, Xte_t = torch.from_numpy(X_tr).to(device), torch.from_numpy(X_te).to(device)

    rows = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(bx)
        sched.step(total / len(y_tr))
        lr = opt.param_groups[0]["lr"]
        done = lr <= args.min_lr or epoch == args.max_epochs
        if epoch == 90 or epoch % args.curve_every == 0 or done:
            tr_mse, tr_scc = snapshot(model, Xtr_t, y_tr, train_df, device)
            te_mse, te_scc = snapshot(model, Xte_t, y_te, test_df, device)
            rows.append({"epoch": epoch, "lr": lr, "train_mse": tr_mse, "test_mse": te_mse,
                         "train_scc": tr_scc, "test_scc": te_scc})
        if done:
            break
    return model, device, rows


def run_config(config, seeds, data, args):
    train_df, test_df, X_tr, y_tr, X_te, y_te, F, near, far = data
    if config == "baseline":
        Xa, Xb = X_tr, X_te
    else:
        mode, fs = config.split("_", 1)
        feats = F(mode, fs)
        tr_pos, te_pos = train_df["_pos"].to_numpy(), test_df["_pos"].to_numpy()
        mu, sd = feats[tr_pos].mean(0), feats[tr_pos].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)                      # constant column -> leave at 0
        z = lambda a: ((a - mu) / sd).astype(np.float32)
        Xa, Xb = np.hstack([X_tr, z(feats[tr_pos])]), np.hstack([X_te, z(feats[te_pos])])
    alleles, thalf = test_df["allele"].to_numpy(), test_df["thalf_hours"].to_numpy()

    runs, per_allele, curves = [], [], []
    for seed in seeds:
        model, device, rows = train_converged(Xa, y_tr, Xb, y_te, train_df, test_df, seed, args)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xb).to(device)).cpu().numpy()
        m, per = evaluate(alleles, thalf, y_te, pred)
        per = per.set_index("allele")["scc"]
        at90 = next((r["test_scc"] for r in rows if r["epoch"] == 90), np.nan)
        last = rows[-1]
        runs.append({"config": config, "seed": seed, "n_inputs": Xa.shape[1],
                     "epochs": last["epoch"], "scc": m["mean_allele_scc"],
                     "scc_near": float(per.reindex(near).mean()),
                     "scc_far": float(per.reindex(far).mean()), "scc_at_epoch_90": at90,
                     "train_scc": last["train_scc"], "train_mse": last["train_mse"],
                     "test_mse": last["test_mse"]})
        per_allele.append(per.rename(seed))
        curves += [{"config": config, "seed": seed, **r} for r in rows]
        print(f"[*] {config:18s} seed {seed:>2} | {last['epoch']:>3} epochs | held-out SCC "
              f"{m['mean_allele_scc']:.3f} (epoch 90: {at90:.3f}) | train SCC "
              f"{last['train_scc']:.3f} | near {runs[-1]['scc_near']:.3f} far "
              f"{runs[-1]['scc_far']:.3f}", flush=True)
    return pd.DataFrame(runs), pd.concat(per_allele, axis=1), pd.DataFrame(curves)


def zero_shot(df, test_pos, y_te, alleles, thalf, features_dir, near, far):
    """Held-out SCC of the raw structural score alone (no training, no labels)."""
    out = []
    for mode in MODES:
        f = load_features(features_dir, mode, "scalar")[test_pos, 0]
        m, per = evaluate(alleles, thalf, y_te, f.astype(np.float64))
        per = per.set_index("allele")["scc"]
        out.append({"score": f"{mode} mean log-prob", "scc": m["mean_allele_scc"],
                    "scc_near": float(per.reindex(near).mean()),
                    "scc_far": float(per.reindex(far).mean())})
    return out


def report(args):
    parts = os.path.join(args.output_dir, "parts")
    runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{parts}/*_runs.csv"))])
    per = {os.path.basename(p)[:-len("_per_allele.csv")]: pd.read_csv(p, index_col=0)
           for p in sorted(glob.glob(f"{parts}/*_per_allele.csv"))}
    g = runs.groupby("config", sort=False)
    tab = g.agg(n_inputs=("n_inputs", "first"), seeds=("seed", "count"),
                epochs=("epochs", "mean"), scc=("scc", "mean"), sd=("scc", "std"),
                scc_near=("scc_near", "mean"), scc_far=("scc_far", "mean"),
                scc_at_90=("scc_at_epoch_90", "mean"), train_scc=("train_scc", "mean"))
    tab = tab.reindex([c for c in ALL_CONFIGS if c in tab.index])
    base = per["baseline"].mean(axis=1) if "baseline" in per else None
    if base is not None:
        d, w = [], []
        for c in tab.index:
            delta = per[c].mean(axis=1) - base
            d.append(delta.mean())
            w.append(int((delta > 0).sum()))
        tab["delta_vs_baseline"] = d
        tab["alleles_improved"] = w
    pd.set_option("display.width", 220)
    print(tab.round(3).to_string())
    tab.to_csv(os.path.join(args.output_dir, "summary.csv"))
    zs = os.path.join(args.output_dir, "zero_shot.json")
    if os.path.exists(zs):
        print("\nzero-shot (no training):")
        print(pd.DataFrame(json.load(open(zs))).round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--features_dir", default="data/inverse_folding")
    p.add_argument("--configs", nargs="+", default=["baseline", "uncond_all", "cond_all",
                                                     "scrambled_all"], choices=ALL_CONFIGS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--curve_every", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output_dir", default="outputs/inverse_folding")
    p.add_argument("--report", action="store_true", help="Only aggregate finished runs")
    args = p.parse_args()
    os.makedirs(os.path.join(args.output_dir, "parts"), exist_ok=True)
    if args.report:
        return report(args)
    torch.set_num_threads(args.threads)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    df["_pos"] = np.arange(len(df))
    train_pos, test_pos = split_by_supertype(df)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    X_tr = encode_sequences(train_df["peptide"].values, train_df["hla_pseudoseq"].values)
    X_te = encode_sequences(test_df["peptide"].values, test_df["hla_pseudoseq"].values)
    y_tr = transform_target(train_df["thalf_hours"].values)
    y_te = transform_target(test_df["thalf_hours"].values)

    # near / far held-out alleles, as in nearest_neighbour.py
    ps = dict(zip(df["allele"], df["hla_pseudoseq"]))
    train_alleles = sorted(train_df["allele"].unique())
    ham = lambda a, b: sum(x != y for x, y in zip(a, b))
    min_ham = {a: min(ham(ps[a], ps[b]) for b in train_alleles) for a in test_df["allele"].unique()}
    near = [a for a, h in min_ham.items() if h <= 2]
    far = [a for a, h in min_ham.items() if h >= 4]

    cache = {}
    def F(mode, fs):
        if (mode, fs) not in cache:
            cache[(mode, fs)] = load_features(args.features_dir, mode, fs)
        return cache[(mode, fs)]

    zs_path = os.path.join(args.output_dir, "zero_shot.json")
    if not os.path.exists(zs_path):
        zs = zero_shot(df, test_pos, y_te, test_df["allele"].to_numpy(),
                       test_df["thalf_hours"].to_numpy(), args.features_dir, near, far)
        json.dump(zs, open(zs_path, "w"), indent=2)
        print("[*] zero-shot structural score alone, held-out alleles:")
        for r in zs:
            print(f"    {r['score']:28s} SCC {r['scc']:.3f} (near {r['scc_near']:.3f}, "
                  f"far {r['scc_far']:.3f})")

    data = (train_df, test_df, X_tr, y_tr, X_te, y_te, F, near, far)
    for config in args.configs:
        runs, per, curves = run_config(config, args.seeds, data, args)
        stem = os.path.join(args.output_dir, "parts", config)
        runs.to_csv(f"{stem}_runs.csv", index=False)
        per.to_csv(f"{stem}_per_allele.csv")
        curves.to_csv(f"{stem}_curves.csv", index=False)
        print(f"[*] {config}: mean held-out SCC {runs['scc'].mean():.3f} +/- "
              f"{runs['scc'].std():.3f} over {len(runs)} seeds, "
              f"{runs['epochs'].mean():.0f} epochs on average", flush=True)


if __name__ == "__main__":
    main()
