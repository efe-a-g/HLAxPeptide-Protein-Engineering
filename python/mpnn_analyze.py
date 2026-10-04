#!/usr/bin/env python3
"""
Does the inverse-folding channel carry allele-specific signal, and is it worth adding?

Reads the mode-(b) matrices from mpnn_score.py and runs, in order, the checks that decide
whether anything downstream is worth building:

  1. SCRAMBLED CONTROL (the gate). Re-score with hla_seq shuffled. If the real and scrambled
     matrices agree, the model is reading backbone geometry alone, every allele gets near
     identical features, and the per-allele framing is decoration. Reported as the correlation
     between real and scrambled, and as how much the matrices vary ACROSS alleles in each --
     a real channel must vary more across alleles than a scrambled one does.
  2. ANCHOR RECOVERY. Known P2 / P-Omega preferences from the literature, against the model's
     top predictions. B*27:05 P2-arginine is the headline case.
  3. PER-POSITION PROFILE. Mean log-prob of the top residue at each position. Anchor chemistry
     implies the model is most constrained at P2 and P9; a flat profile means it is not
     capturing anchors.
  4. ZERO-SHOT. Score every held-out peptide by summing its residues' entries in its allele's
     matrix, and take the mean per-allele Spearman -- no training, no labels. This is row 1 of
     the three-row table; rows 2 and 3 are the trained baseline and baseline+features.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from baseline import evaluate, split_by_supertype, transform_target

AA20 = list("ARNDCQEGHILKMFPSTWYV")
IDX = {a: i for i, a in enumerate(AA20)}

# Canonical primary anchors from the class I literature (Sidney et al. 2008 supertype motifs).
KNOWN = {
    "HLA-A*01:01": ("TS", "Y"), "HLA-A*02:01": ("LM", "VL"), "HLA-A*03:01": ("LVM", "KY"),
    "HLA-A*11:01": ("VTLM", "K"), "HLA-A*24:02": ("YF", "FLI"), "HLA-B*07:02": ("P", "LF"),
    "HLA-B*08:01": ("K", "LF"), "HLA-B*15:01": ("LQM", "FY"), "HLA-B*27:05": ("R", "KRLF"),
    "HLA-B*35:01": ("P", "YFMLI"), "HLA-B*44:05": ("E", "FWY"), "HLA-B*51:01": ("APG", "IV"),
    "HLA-B*57:01": ("ATS", "WF"), "HLA-B*58:01": ("ATS", "WF"),
}


def load(path):
    z = np.load(path, allow_pickle=True)
    df = pd.DataFrame({"template": z["template"], "allele": z["allele"]})
    return z["pssm"], df


def mean_over_templates(pssm, meta):
    """Average the per-template matrices for each allele -> {allele: (9, 20)}."""
    out = {}
    for allele, g in meta.groupby("allele"):
        out[allele] = pssm[g.index.to_numpy()].mean(0)
    return out


def across_allele_spread(mats):
    """Mean SD across alleles of each (position, aa) entry. Zero => all alleles identical."""
    stack = np.stack([mats[a] for a in sorted(mats)])
    return float(stack.std(0).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--pssm", default="outputs/mpnn/uncond_pssm.npz")
    p.add_argument("--scrambled", default="outputs/mpnn/uncond_pssm_scrambled.npz")
    p.add_argument("--output_dir", default="outputs/mpnn")
    args = p.parse_args()

    pssm, meta = load(args.pssm)
    mats = mean_over_templates(pssm, meta)
    print(f"[*] {pssm.shape[0]} allele-template matrices over "
          f"{meta['template'].nunique()} templates, {len(mats)} alleles\n")

    report = {}

    # 1. scrambled control -------------------------------------------------------------
    print("=" * 74)
    print("1. SCRAMBLED-HLA CONTROL  (the gate)")
    if os.path.isfile(args.scrambled):
        s_pssm, s_meta = load(args.scrambled)
        s_mats = mean_over_templates(s_pssm, s_meta)
        common = sorted(set(mats) & set(s_mats))
        r = np.corrcoef(np.stack([mats[a].ravel() for a in common]).ravel(),
                        np.stack([s_mats[a].ravel() for a in common]).ravel())[0, 1]
        real_spread, scram_spread = across_allele_spread(mats), across_allele_spread(s_mats)
        report["scrambled"] = {"corr_real_vs_scrambled": float(r),
                               "across_allele_spread_real": real_spread,
                               "across_allele_spread_scrambled": scram_spread,
                               "spread_ratio": real_spread / max(scram_spread, 1e-9)}
        print(f"   corr(real, scrambled) over all entries : {r:+.4f}")
        print(f"   across-allele spread, real             : {real_spread:.4f}")
        print(f"   across-allele spread, scrambled        : {scram_spread:.4f}")
        print(f"   ratio (want >> 1)                      : {real_spread/max(scram_spread,1e-9):.2f}")
    else:
        print(f"   [!] {args.scrambled} not found -- control not run")

    # 2. anchor recovery ---------------------------------------------------------------
    print("=" * 74)
    print("2. ANCHOR RECOVERY  (top-3 predicted vs literature)")
    hits2 = hits9 = n = 0
    rows = []
    for allele, (p2, p9) in KNOWN.items():
        if allele not in mats:
            continue
        m = mats[allele]
        t2 = "".join(np.array(AA20)[np.argsort(-m[1])[:3]])
        t9 = "".join(np.array(AA20)[np.argsort(-m[8])[:3]])
        ok2, ok9 = any(c in p2 for c in t2[:1]), any(c in p9 for c in t9[:1])
        hits2 += ok2; hits9 += ok9; n += 1
        rows.append({"allele": allele, "P2_pred": t2, "P2_known": p2, "P2_top1_hit": ok2,
                     "P9_pred": t9, "P9_known": p9, "P9_top1_hit": ok9})
        print(f"   {allele:14s} P2 {t2} (known {p2:5s}) {'OK ' if ok2 else '   '} | "
              f"P9 {t9} (known {p9:5s}) {'OK' if ok9 else ''}")
    print(f"   top-1 anchor hit rate: P2 {hits2}/{n}   P-Omega {hits9}/{n}")
    report["anchor_recovery"] = {"p2_hits": hits2, "p9_hits": hits9, "n": n}
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "anchor_recovery.csv"), index=False)

    # 3. per-position profile ----------------------------------------------------------
    print("=" * 74)
    print("3. PER-POSITION PROFILE  (expect most constrained at P2 and P9)")
    stack = np.stack([mats[a] for a in sorted(mats)])
    top_lp = stack.max(2).mean(0)
    ent = (-(np.exp(stack) * stack).sum(2)).mean(0)
    for i in range(9):
        bar = "#" * int(round((top_lp[i] - top_lp.min()) / max(float(top_lp.max() - top_lp.min()), 1e-9) * 40))
        print(f"   P{i+1}  top log p {top_lp[i]:+.3f}  entropy {ent[i]:.3f}  {bar}")
    report["per_position"] = {"top_logp": top_lp.tolist(), "entropy": ent.tolist()}

    # 4. zero-shot ---------------------------------------------------------------------
    print("=" * 74)
    print("4. ZERO-SHOT  (sum the peptide's entries; no training, no labels)")
    df = pd.read_csv(args.data_path).reset_index(drop=True)
    _, test_pos = split_by_supertype(df, verbose=False)
    te = df.iloc[test_pos]
    s_te = transform_target(te["thalf_hours"].values)
    pred = np.array([mats[a][np.arange(9), [IDX[c] for c in pep]].sum()
                     if a in mats else np.nan
                     for a, pep in zip(te["allele"], te["peptide"])])
    ok = ~np.isnan(pred)
    m, per = evaluate(te["allele"].to_numpy()[ok], None, s_te[ok], pred[ok])
    print(f"   mean per-allele SCC : {m['mean_allele_scc']:+.4f}   "
          f"({m['n_alleles_scored']} alleles)")
    print(f"   global SCC          : {m['global_scc']:+.4f}")
    print(f"   for reference: peptide-only null 0.2950, trained baseline 0.4219 (ensemble)")
    report["zero_shot"] = {k: m[k] for k in ("mean_allele_scc", "global_scc", "n_alleles_scored")}
    print(per.to_string(index=False))
    per.to_csv(os.path.join(args.output_dir, "zero_shot_per_allele.csv"), index=False)

    np.savez(os.path.join(args.output_dir, "pssm_by_allele.npz"),
             allele=np.array(sorted(mats)), pssm=np.stack([mats[a] for a in sorted(mats)]),
             aa_order=np.array(AA20))
    with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 74)
    print(f"[*] wrote pssm_by_allele.npz, analysis.json to {args.output_dir}/")


if __name__ == "__main__":
    main()
