#!/usr/bin/env python3
"""Create publication-quality descriptive figures from Phase 0 audit artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

OUT = Path("outputs/figures")
BLUE, RED, GREEN, GRAY = "#176B91", "#C85245", "#287F6C", "#87929D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"audit_{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def target_distribution(data, audit):
    epsilon = audit["config"]["epsilon_hours"]
    y = data.thalf_hours.to_numpy()
    positive = y[y > 0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [0.7, 1.3, 1.3]})
    ax = axes[0]
    counts = [(y == 0).sum(), len(positive)]
    ax.bar([0, 1], counts, color=[RED, BLUE], width=.64)
    for i, count in enumerate(counts):
        ax.text(i, count + 450, f"{count:,}\n({count / len(y):.1%})", ha="center", fontsize=10)
    ax.set(xticks=[0, 1], xticklabels=["Exactly zero", "Positive"], ylabel="Rows", ylim=(0, max(counts) * 1.23), title="A  |  Point mass at zero")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax = axes[1]
    bins = np.geomspace(positive.min() * .95, positive.max() * 1.05, 46)
    ax.hist(positive, bins=bins, color=BLUE, edgecolor="white", linewidth=.45)
    ax.set(xscale="log", xlabel="Positive half-life (hours; logarithmic scale)", ylabel="Rows", title="B  |  Positive observations")
    ax.axvline(2, linestyle=":", color=GREEN, linewidth=1.5, label="Stability cutoff: > 2 h")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[2]
    transformed = np.log(y + epsilon)
    floor = np.log(epsilon)
    bins = np.linspace(floor - .13, transformed.max() + .13, 48)
    ax.hist(transformed[y > 0], bins=bins, color=BLUE, edgecolor="white", linewidth=.4, label="Positive")
    ax.hist(transformed[y == 0], bins=bins, color=RED, edgecolor="white", linewidth=.4, label="Exactly zero")
    ax.axvline(floor, color="#723C37", linewidth=1.3, linestyle="--")
    ax.annotate(f"Chosen transform floor\nlog({epsilon}) = {floor:.2f}", xy=(floor, counts[0]),
                xytext=(floor + 1.2, counts[0] * .91), fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#723C37"}, color="#723C37")
    ax.set(xlabel=f"log(half-life + {epsilon})", ylabel="Rows", title="C  |  Model target", ylim=(0, counts[0] * 1.2))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Peptide–HLA stability: 28,166 observations across 75 alleles", fontsize=15, x=.07, ha="left")
    fig.text(.07, -.015, "ε = 0.05 h is a chosen numerical offset, not an established assay detection threshold. Zeros alone do not prove synthetic padding.", fontsize=9, color="#4B5560")
    fig.tight_layout(rect=[0, .04, 1, .93])
    save(fig, "target_distribution")


def allele_distribution(summary):
    summary = summary.sort_values("n", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 15), sharey=True, gridspec_kw={"width_ratios": [1.45, 1]})
    positions = np.arange(len(summary))
    colors = [RED if n < 50 else BLUE for n in summary.n]
    axes[0].barh(positions, summary.n, color=colors, height=.77)
    axes[0].set(yticks=positions, yticklabels=summary.allele.str.replace("HLA-", "", regex=False), xlabel="Number of rows", title="A  |  Allele representation")
    axes[0].tick_params(axis="y", labelsize=7)
    for i in [0, 6, len(summary) - 2, len(summary) - 1]:
        axes[0].text(summary.loc[i, "n"] + 8, i, f"{summary.loc[i, 'n']:,}", va="center", fontsize=8)
    axes[0].set_xlim(0, 1175)
    axes[0].text(.98, .30, "7 alleles have < 50 rows\nLargest/smallest count: 153×", transform=axes[0].transAxes,
                 ha="right", color=RED, fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95})
    axes[1].barh(positions, summary.zero_fraction, color=RED, height=.77)
    axes[1].axvline(summary.n_zero.sum() / summary.n.sum(), color="#555555", linewidth=1.3, linestyle="--", label="Pooled zero fraction: 20.2%")
    axes[1].set(xlim=(0, 1), xlabel="Fraction of rows exactly zero", title="B  |  Zero prevalence varies by allele")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    extreme = summary.zero_fraction.idxmax()
    axes[1].text(.915, extreme + 1.3, "B*39:06(C67S): 92.1%", ha="right", fontsize=9, color="#8E362E")
    for ax in axes:
        ax.set_ylim(-1, len(summary))
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle("Allele imbalance and zero clustering", fontsize=16, x=.15, ha="left", y=.985)
    fig.text(.15, .008, "Alleles sorted by sample count. Within-allele metrics are essential; pooled metrics mix distinct allele distributions.", fontsize=9, color="#4B5560")
    fig.tight_layout(rect=[0, .025, 1, .972])
    save(fig, "allele_distribution")


def empirical_motifs(motifs):
    alleles = ["HLA-A*02:01", "HLA-B*27:05", "HLA-A*03:01"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.7), sharey=True)
    for ax, allele in zip(axes, alleles):
        values = np.asarray(motifs["alleles"][allele]["probabilities"]).T
        im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        n = motifs["alleles"][allele]["n_stable"]
        ax.set(title=f"{allele}\n{n:,} peptides with half-life > 2 h", xticks=np.arange(9),
               xticklabels=[f"P{i}" for i in range(1, 10)], yticks=np.arange(20),
               yticklabels=list(motifs["amino_acids"]), xlabel="Peptide position")
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, 9), minor=True)
        ax.set_yticks(np.arange(-.5, 20), minor=True)
        ax.grid(which="minor", color="white", linewidth=.4)
        ax.tick_params(which="minor", length=0)
        for position in [1, 8]:
            ax.add_patch(plt.Rectangle((position - .5, -.5), 1, 20, fill=False, edgecolor="#D49F32", linewidth=1.5))
            ax.get_xticklabels()[position].set_fontweight("bold")
        for aa, position in zip(*np.where(values >= .2)):
            v = values[aa, position]
            ax.text(position, aa, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > .5 else "#163644")
    axes[0].set_ylabel("Amino acid")
    fig.suptitle("Empirical stable-peptide motifs recover familiar anchor preferences", fontsize=15, x=.07, ha="left")
    fig.subplots_adjust(top=.86, bottom=.12, left=.07, right=.90, wspace=.15)
    cax = fig.add_axes([.92, .15, .016, .65])
    fig.colorbar(im, cax=cax, label="Position-specific probability")
    fig.text(.07, .025, "Full-data descriptive motifs; pseudocount 0.5 per amino acid per position. Gold boxes mark P2 and P9. Not used as supervised test features.", fontsize=9, color="#4B5560")
    save(fig, "empirical_motifs")


def wilson(count, n):
    z = 1.96
    p = count / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-radius, center+radius


def zero_anchor_composition(audit):
    specs = [("HLA-A*02:01_P2_LM", "A*02:01 · P2 L/M"), ("HLA-A*02:01_P9_VL", "A*02:01 · P9 V/L"),
             ("HLA-B*27:05_P2_R", "B*27:05 · P2 R"), ("HLA-A*03:01_P9_KR", "A*03:01 · P9 K/R")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for ax, (key, title) in zip(axes.flat, specs):
        check = audit["motif_biology_checks"][key]
        groups = [check["zero"], check["positive"], {"n": check["n_stable"], "preferred_count": check["preferred_count"], "preferred_fraction": check["preferred_fraction_unsmoothed"]}]
        values = [g["preferred_fraction"] for g in groups]
        ax.bar(range(3), values, color=[RED, BLUE, GREEN], width=.6)
        for i, (g, v) in enumerate(zip(groups, values)):
            low, high = wilson(g["preferred_count"], g["n"])
            ax.errorbar(i, v, yerr=[[v-low], [high-v]], color="#303A42", capsize=3, linewidth=1.2)
            ax.text(i, high + .035, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
        ax.set(title=title, xticks=range(3), xticklabels=[f"Zero\nn = {groups[0]['n']}", f"Positive\nn = {groups[1]['n']}", f"> 2 h\nn = {groups[2]['n']}"], ylim=(0, 1.15), ylabel="Fraction with preferred anchor")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Zero rows are not uniformly depleted of preferred anchors", fontsize=15, x=.08, ha="left")
    fig.text(.08, .01, "Bars: unsmoothed frequencies; whiskers: descriptive 95% Wilson intervals. The > 2 h group is a subset of positive rows.\nZero-row composition and allele clustering do not establish whether individual zeros were measured or assigned.", fontsize=9, color="#4B5560")
    fig.tight_layout(rect=[0, .065, 1, .94], h_pad=2)
    save(fig, "zero_anchor_composition")


def main():
    audit = json.loads(Path("outputs/audit/audit.json").read_text())
    motifs = json.loads(Path("outputs/audit/empirical_motifs.json").read_text())
    data = pd.read_csv(audit["source"])
    summary = pd.read_csv("outputs/audit/allele_summary.csv")
    target_distribution(data, audit)
    allele_distribution(summary)
    empirical_motifs(motifs)
    zero_anchor_composition(audit)
    print("Wrote four audit figures as PNG and vector PDF in outputs/figures.")


if __name__ == "__main__":
    main()
