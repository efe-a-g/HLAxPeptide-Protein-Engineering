#!/usr/bin/env python3
"""Diagram of the baseline network (baseline.py: StabilityNet) for slides.

  python python/plot_architecture.py   # writes outputs/baseline_architecture.png (16:9, 300 dpi)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#d9d8d3"
BLUE_BG, ORANGE_BG, GREY_BG = "#e6effb", "#fdece5", "#f1f0ed"


def box(ax, x, y, w, h, title, sub=None, edge=INK2, fill=GREY_BG, lw=1.6, tsize=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=edge, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=tsize,
                color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=tsize - 3.2,
                color=INK2, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize,
                color=INK, weight="bold")


def arrow(ax, p0, p1, color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, lw=1.8,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    ax.text(0.5, 8.4, "Baseline: pan-specific peptide–HLA stability network", fontsize=22,
            color=INK, weight="bold", va="center")
    ax.text(0.5, 7.9, "Single hidden layer, sigmoid units  ·  after NetMHCstabpan "
            "(Rasmussen et al. 2016)  ·  51,721 trainable parameters", fontsize=12.5,
            color=INK2, va="center")

    # inputs
    box(ax, 0.5, 5.25, 3.0, 1.55, "Peptide", "9-mer\ne.g. VTTEVAFGL", edge=BLUE, fill=BLUE_BG)
    box(ax, 0.5, 2.05, 3.0, 1.55, "HLA pseudosequence", "34 pocket residues\ne.g. YFAMYGEKVAHTHV…",
        edge=ORANGE, fill=ORANGE_BG, tsize=12.5)

    # encoding
    box(ax, 4.45, 5.25, 2.5, 1.55, "BLOSUM50", "9 × 20 = 180", edge=BLUE, fill=BLUE_BG)
    box(ax, 4.45, 2.05, 2.5, 1.55, "BLOSUM50", "34 × 20 = 680", edge=ORANGE, fill=ORANGE_BG)
    arrow(ax, (3.5, 6.02), (4.45, 6.02), BLUE)
    arrow(ax, (3.5, 2.82), (4.45, 2.82), ORANGE)
    ax.text(5.7, 7.1, "encode each residue as\nits 20-value BLOSUM50 row", ha="center", fontsize=10.5,
            color=MUTED, va="center", linespacing=1.3)

    # concat
    box(ax, 7.9, 3.65, 1.5, 1.7, "Concat", "860\ninputs", tsize=13)
    arrow(ax, (6.95, 6.02), (7.9, 4.95), BLUE, rad=-0.12)
    arrow(ax, (6.95, 2.82), (7.9, 4.05), ORANGE, rad=0.12)

    # hidden
    box(ax, 10.2, 3.65, 2.1, 1.7, "Hidden layer", "Linear 860 → 60\nsigmoid", tsize=13)
    arrow(ax, (9.4, 4.5), (10.2, 4.5))

    # output
    box(ax, 13.05, 3.65, 2.1, 1.7, "Output", "Linear 60 → 1\nsigmoid", tsize=13)
    arrow(ax, (12.3, 4.5), (13.05, 4.5))
    ax.text(13.85, 2.55, "predicted stability\nŝ ∈ [0, 1]", ha="right", linespacing=1.3, fontsize=12, color=INK,
            weight="bold", va="center")

    # loss / target strip
    ax.add_patch(FancyBboxPatch((0.5, 0.35), 15.0, 1.1, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=GREY_BG, ec=NEUTRAL, lw=1.2))
    ax.text(0.8, 0.9, "Training target", fontsize=11.5, color=INK, weight="bold", va="center")
    ax.text(3.0, 0.9, "s = 2^(−1 h / t½)     (t½ = 0 → s = 0)", fontsize=12, color=INK2,
            va="center")
    ax.text(7.6, 0.9, "Loss", fontsize=11.5, color=INK, weight="bold", va="center")
    ax.text(8.5, 0.9, "MSE(ŝ, s)", fontsize=12, color=INK2, va="center")
    ax.text(10.4, 0.9, "Optimiser", fontsize=11.5, color=INK, weight="bold", va="center")
    ax.text(11.7, 0.9, "Adam, lr 1e-3, batch 128", fontsize=12, color=INK2, va="center")
    arrow(ax, (14.1, 3.65), (14.1, 1.45), MUTED)

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/baseline_architecture.png"
    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches=None)
    print(f"[*] Wrote {out}")


if __name__ == "__main__":
    main()
