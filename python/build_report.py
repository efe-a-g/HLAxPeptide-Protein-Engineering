#!/usr/bin/env python3
"""
Build the self-contained local HTML report in outputs/report/.

Reads outputs/ablation/results.jsonl, the per-allele CSVs, and (if present)
outputs/report/structures.json, and emits index.html with all data embedded as
JSON. Charts are hand-built SVG drawn by vanilla JS; 3D viewers use NGL from a
CDN. Nothing is published anywhere -- open the file locally.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from aggregate import METRICS, PRETTY, SPLIT_ORDER, pretty

BASELINE = "BLOSUM-pep + BLOSUM-HLA"

# Which family each arm belongs to -> drives colour (3 validated slots).
def group_of(pep, hla):
    learned_pep = pep.startswith("esm")
    learned_hla = hla.startswith(("boltz", "esm"))
    if pep == "none" or hla == "none":
        return "ablation"
    if pep == "blosum_pep" and hla == "blosum_hla":
        return "baseline"
    if hla == "onehot_hla":
        return "control"
    if learned_pep or learned_hla:
        return "foundation"
    return "control"


SPLIT_META = {
    "random": {
        "name": "Random split",
        "blurb": "Rows shuffled. Every test allele and ~49% of test peptides also "
                 "appear in training. The easiest split, and the least informative: "
                 "a per-allele vector can act as a memorised lookup key here.",
    },
    "cluster": {
        "name": "Peptide-grouped split",
        "blurb": "No peptide appears in both train and test (verified: 0 shared). "
                 "Tests generalisation to unseen peptides; all alleles are still seen.",
    },
    "allele": {
        "name": "Leave-allele-out split",
        "blurb": "15 held-out alleles, zero allele overlap (verified). This is the "
                 "real test of the premise: a per-allele HLA embedding can only help "
                 "if it generalises to alleles never seen in training.",
    },
}


def build_payload(results_path, noise_path, struct_path):
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    df = pd.DataFrame(rows)
    df["config"] = [pretty(p, h) for p, h in zip(df["pep"], df["hla"])]
    df["group"] = [group_of(p, h) for p, h in zip(df["pep"], df["hla"])]
    df = df.drop_duplicates(["config", "split", "seed", "tag", "pca_dim"],
                            keep="last")

    # ---- summary: mean / std per (config, split) --------------------------
    summary = []
    for (cfg, split), g in df.groupby(["config", "split"]):
        rec = {"config": cfg, "split": split, "group": g["group"].iloc[0],
               "n_features": int(g["n_features"].iloc[0]),
               "n_seeds": int(len(g)),
               "runtime_s": float(g["runtime_s"].mean())}
        for m in METRICS:
            rec[m] = float(g[m].mean())
            rec[m + "_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        summary.append(rec)

    # ---- paired per-seed deltas vs baseline -------------------------------
    deltas = []
    for split, sub in df.groupby("split"):
        base = sub[sub["config"] == BASELINE].set_index("seed")
        if base.empty:
            continue
        for cfg, g in sub.groupby("config"):
            if cfg == BASELINE:
                continue
            g = g.set_index("seed")
            shared = sorted(set(g.index) & set(base.index))
            if not shared:
                continue
            rec = {"config": cfg, "split": split, "n_paired": len(shared),
                   "group": g["group"].iloc[0]}
            for m in METRICS:
                d = g.loc[shared, m].to_numpy() - base.loc[shared, m].to_numpy()
                sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
                se = sd / np.sqrt(len(d)) if len(d) > 1 else 0.0
                rec[f"d_{m}"] = float(np.mean(d))
                rec[f"d_{m}_sd"] = sd
                rec[f"d_{m}_t"] = float(np.mean(d) / se) if se > 1e-12 else 0.0
            deltas.append(rec)

    # ---- per-allele: baseline vs the best foundation arm ------------------
    def slugify(cfg):
        return cfg
    inv = {v: k for k, v in PRETTY.items()}

    def load_per_allele(cfg, split):
        pep_h = cfg.split(" + ")
        pep, hla = inv.get(pep_h[0], pep_h[0]), inv.get(pep_h[1], pep_h[1])
        frames = []
        for p in glob.glob(f"outputs/ablation/per_allele/{pep}__{hla}__{split}__s*.csv"):
            if re.search(r"__s\d+__", os.path.basename(p)):
                continue  # tagged variant run, not the main grid
            t = pd.read_csv(p)
            if len(t):
                frames.append(t)
        if not frames:
            return None
        allc = pd.concat(frames)
        out = allc.groupby("allele").agg(
            n_samples=("n_samples", "mean"), pcc=("pcc", "mean"),
            pcc_sd=("pcc", "std"), n_seeds=("pcc", "size")).reset_index()
        return out

    per_allele = {}
    for split in SPLIT_ORDER:
        s = [r for r in summary if r["split"] == split]
        if not s:
            continue
        fm = [r for r in s if r["group"] == "foundation"]
        best_fm = max(fm, key=lambda r: r["global_pcc_score"])["config"] if fm else None
        b = load_per_allele(BASELINE, split)
        f = load_per_allele(best_fm, split) if best_fm else None
        if b is None:
            continue
        merged = b.rename(columns={"pcc": "pcc_base", "pcc_sd": "pcc_base_sd"})
        if f is not None:
            merged = merged.merge(
                f[["allele", "pcc", "pcc_sd"]].rename(
                    columns={"pcc": "pcc_fm", "pcc_sd": "pcc_fm_sd"}),
                on="allele", how="outer")
        per_allele[split] = {
            "baseline": BASELINE, "foundation": best_fm,
            "rows": json.loads(merged.to_json(orient="records")),
        }

    structures = []
    if os.path.isfile(struct_path):
        structures = json.load(open(struct_path))

    noise = json.load(open(noise_path)) if os.path.isfile(noise_path) else {}

    return {
        "summary": summary, "deltas": deltas, "per_allele": per_allele,
        "runs": json.loads(df.to_json(orient="records")),
        "structures": structures, "noise": noise,
        "baseline": BASELINE,
        "split_meta": SPLIT_META, "split_order": SPLIT_ORDER,
        "metrics": METRICS,
        "seeds": sorted(int(s) for s in df["seed"].unique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    ap.add_argument("--noise", default="outputs/ablation/noise_floor.json")
    ap.add_argument("--structures", default="outputs/report/structures.json")
    ap.add_argument("--out", default="outputs/report/index.html")
    ap.add_argument("--template", default="python/report_template.html")
    args = ap.parse_args()

    payload = build_payload(args.results, args.noise, args.structures)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tpl = open(args.template, encoding="utf-8").read()
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False,
                      default=lambda o: None)
    html = tpl.replace("/*__DATA__*/", blob)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    kb = os.path.getsize(args.out) / 1024
    print(f"[+] wrote {args.out} ({kb:.0f} KB) | "
          f"{len(payload['summary'])} summary rows, "
          f"{len(payload['deltas'])} deltas, "
          f"{len(payload['structures'])} structures")


if __name__ == "__main__":
    main()
