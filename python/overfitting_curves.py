#!/usr/bin/env python3
"""
Train-vs-held-out curves for the baseline network, to show how it overfits.

Trains baseline.py's StabilityNet exactly as the baseline does (BLOSUM50 features, MSE on
s = 2^(-t0/t_half), Adam lr 1e-3, weight decay 1e-5, ReduceLROnPlateau on the running train
loss, batch size 128) but for many more epochs than the baseline's 90, and every few epochs
records, with the model in eval mode:

  * mean per-allele Spearman on the *training* alleles (in-sample) and on the held-out alleles
  * MSE on the training rows and on the held-out rows

Held-out alleles are the project-wide split from baseline.split_by_supertype. The held-out
curves are for diagnosis only; no epoch is chosen with them.

  python python/overfitting_curves.py                 # train (slow) and plot
  python python/overfitting_curves.py --plot_only     # re-plot from outputs/overfitting/curves.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import (StabilityNet, encode_sequences, evaluate, resolve_device,
                      split_by_supertype, transform_target)


def snapshot(model, X, y, d, device):
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    m, _ = evaluate(d["allele"].to_numpy(), d["thalf_hours"].to_numpy(), y, pred)
    return float(np.mean((pred - y) ** 2)), m["mean_allele_scc"]


def train_curve(seed, data, args):
    train_df, test_df, X_tr, y_tr, X_te, y_te = data
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
    for epoch in range(1, args.epochs + 1):
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
        if epoch == 1 or epoch % args.eval_every == 0:
            tr_mse, tr_scc = snapshot(model, Xtr_t, y_tr, train_df, device)
            te_mse, te_scc = snapshot(model, Xte_t, y_te, test_df, device)
            rows.append({"seed": seed, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                         "train_mse": tr_mse, "test_mse": te_mse,
                         "train_scc": tr_scc, "test_scc": te_scc})
    return rows


def plot(curves, path, baseline_epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Reference palette (light mode): categorical slots 1 and 2, text and surface tokens.
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    TRAIN, TEST = "#2a78d6", "#eb6834"

    g = curves.groupby("epoch")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    panels = [("scc", "Spearman rank correlation (mean over alleles)",
               "Ranking: training keeps improving, held-out plateaus", (0, 1.0)),
              ("mse", "Mean squared error",
               "Error: held-out turns upward, training keeps falling", (0, None))]
    for ax, (metric, ylabel, title, ylim) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for col, colour in ((f"train_{metric}", TRAIN), (f"test_{metric}", TEST)):
            mean, lo, hi = g[col].mean(), g[col].min(), g[col].max()
            ax.fill_between(mean.index, lo, hi, color=colour, alpha=0.10, lw=0)
            ax.plot(mean.index, mean.values, color=colour, lw=1.8, solid_capstyle="round")
            ax.plot(mean.index[-1], mean.values[-1], "o", ms=8, color=colour,
                    mec=SURFACE, mew=2, zorder=5)
            ax.annotate(f"{mean.values[-1]:.2f}" if metric == "scc" else f"{mean.values[-1]:.3f}",
                        (mean.index[-1], mean.values[-1]), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.axvline(baseline_epochs, color=MUTED, lw=1, ls=(0, (1, 0)), zorder=1)
        ax.text(baseline_epochs + 6, 0.02, f"baseline stops\nat epoch {baseline_epochs}",
                transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED, va="bottom")
        if metric == "mse":
            best = int(g["test_mse"].mean().idxmin())
            ax.plot(best, g["test_mse"].mean().loc[best], "o", ms=8, color=TEST,
                    mec=SURFACE, mew=2, zorder=6)
            ax.annotate(f"lowest held-out error\n(epoch {best})",
                        (best, g["test_mse"].mean().loc[best]), xytext=(14, -34),
                        textcoords="offset points", fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_ylim(*ylim)
        ax.set_xlim(0, curves["epoch"].max() * 1.06)
        ax.set_xlabel("training epoch", fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10, weight="bold")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=8.5)

    n_seeds = curves["seed"].nunique()
    handles = [Line2D([0], [0], color=TRAIN, lw=2, label="training alleles (seen in training)"),
               Line2D([0], [0], color=TEST, lw=2, label="held-out alleles (never seen)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995), labelcolor=INK2)
    fig.text(0.005, 0.005, f"Baseline network, {curves['epoch'].max()} epochs, mean of {n_seeds} "
             "seeds (band = min to max across seeds).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[*] Wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--baseline_epochs", type=int, default=90,
                   help="Where baseline.py stops, drawn as a reference line")
    p.add_argument("--output_dir", default="outputs/overfitting")
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "curves.csv")

    if not args.plot_only:
        df = pd.read_csv(args.data_path).reset_index(drop=True)
        train_pos, test_pos = split_by_supertype(df)
        train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
        X_tr = encode_sequences(train_df["peptide"].values, train_df["hla_pseudoseq"].values)
        X_te = encode_sequences(test_df["peptide"].values, test_df["hla_pseudoseq"].values)
        y_tr = transform_target(train_df["thalf_hours"].values)
        y_te = transform_target(test_df["thalf_hours"].values)
        data = (train_df, test_df, X_tr, y_tr, X_te, y_te)
        rows = []
        for seed in args.seeds:
            rows += train_curve(seed, data, args)
            pd.DataFrame(rows).to_csv(csv_path, index=False)   # keep partial progress
            last = rows[-1]
            print(f"[*] seed {seed} done | final train SCC {last['train_scc']:.3f} "
                  f"held-out SCC {last['test_scc']:.3f} | train MSE {last['train_mse']:.4f} "
                  f"held-out MSE {last['test_mse']:.4f}", flush=True)

    plot(pd.read_csv(csv_path), os.path.join(args.output_dir, "overfitting.png"),
         args.baseline_epochs)


if __name__ == "__main__":
    main()
