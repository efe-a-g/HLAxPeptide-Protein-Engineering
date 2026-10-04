
import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_figure_style(sizes=(16, 14, 13), frame="open")
mpl.rcParams["svg.fonttype"] = "path"

FM = "repo/foundation-models-testing/HLAxPeptide-Protein-Engineering-foundation-models-testing/"
CF = "repo/confidence/HLAxPeptide-Protein-Engineering-confidence/confidence/"
A_COL, B_COL = "#2E6E9E", "#C07A28"
REF_B, REF_N, NEUTRAL = "#2B2B2B", "#8A8A8A", "#C4C4C4"

sup = pd.DataFrame([json.loads(l) for l in open(FM+"outputs/supertype/results.jsonl", encoding="utf-8")])
agg = sup.groupby("label")["mean_allele_scc"].agg(["mean","std"]).reset_index()
get = lambda lb: (float(agg.loc[agg.label==lb,"mean"].iloc[0]), float(agg.loc[agg.label==lb,"std"].iloc[0]))
BASE, BASE_SD = get("blosum_pep | blosum_hla")
NULL, NULL_SD = get("peptide-only null")
A_BEST, A_BEST_SD = get("esm2_150m_pep_res | boltz_BF")
pc = pd.read_csv(CF+"outputs/predictAll_predictor_comparison.csv")
rb = pc.loc[pc.mean_allele_scc.idxmax()]
B_BEST, B_BEST_SE = float(rb.mean_allele_scc), float(rb.se)

# ----------------------------------------------------------------- figure 2: benchmark
fig2 = plt.figure(figsize=(12.0, 5.0))
ax2 = fig2.add_axes([0.34, 0.17, 0.60, 0.74])
rows = [("Route B best\nco-folding confidence", B_BEST, B_BEST_SE, B_COL),
        ("no HLA at all\npeptide sequence only", NULL, NULL_SD, NEUTRAL),
        ("Route A best\npocket embedding", A_BEST, A_BEST_SD, A_COL),
        ("what we already have\nBLOSUM pseudosequence", BASE, BASE_SD, REF_B)]
y = np.arange(len(rows))
ax2.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.6,
         color=[r[3] for r in rows], error_kw=dict(ecolor="#3A3A3A", elinewidth=1.6, capsize=4), zorder=3)
for yi, r in zip(y, rows):
    ax2.text(r[1]+0.012, yi, "%.2f" % r[1], va="center", ha="left", fontsize=17, weight="bold", color=r[3])
ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=15)
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.set_xlim(0, 0.56); ax2.set_ylim(-0.6, len(rows)-0.4)
ax2.set_xlabel("how well it ranks peptides within an allele   (higher = better)", fontsize=15)
fig2.savefig("fig_benchmark.svg")

# ---------------------------------------------------------------- figure 3: shrinkage
fig3 = plt.figure(figsize=(7.4, 4.8))
ax3 = fig3.add_axes([0.17, 0.17, 0.60, 0.74])
x = [0,1,2]; v = [0.289, 0.178, 0.095]
ax3.plot(x, v, "-o", color=B_COL, lw=3.0, ms=10, zorder=3)
for xi, vi in zip(x, v):
    ax3.text(xi, vi+0.016, "%.2f" % vi, ha="center", fontsize=15, weight="bold", color=B_COL)
ax3.axhline(NULL, color=REF_N, lw=1.8, ls=(0,(4,3)), zorder=1)
ax3.text(1.0, NULL+0.014, "score with no HLA information at all", fontsize=14, color="#5E5E5E", ha="center")
ax3.set_xticks(x); ax3.set_xticklabels(["6\nalleles","14\nalleles","67\nalleles"], fontsize=15)
ax3.set_yticks([0.1,0.2,0.3]); ax3.set_ylim(0.03, 0.355); ax3.set_xlim(-0.12, 2.3)
ax3.set_ylabel("how well it ranks peptides", fontsize=15)
ax3.text(2.12, 0.095, "measured on\nevery allele", fontsize=14, color=B_COL, va="center", linespacing=1.4)
fig3.savefig("fig_shrinkage.svg")
print("BASE %.4f NULL %.4f A_BEST %.4f B_BEST %.4f (%s)" % (BASE, NULL, A_BEST, B_BEST, rb.predictor))
