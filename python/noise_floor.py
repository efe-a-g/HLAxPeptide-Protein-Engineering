#!/usr/bin/env python3
"""
Quantify the noise floor three ways and write outputs/ablation/noise_floor.json.

  unseeded  Repeats of the published train_baseline.py at a FIXED seed. It seeds
            numpy but never torch, so these runs differ only by uncontrolled
            weight init and batch shuffling. Hard-coded from the three logged
            reproduction runs (outputs/repro/).
  init      Harness runs with the split FIXED (--seed 42) and --torch_seed
            varied. The controlled version of the above.
  split     Harness runs with the seed varied, which moves both the split draw
            and the init. This is the spread the +- figures on this project use,
            and it is the one that matters for comparing arms.

The point of separating them: if split variance dominates init variance, then
comparing arms *unpaired* across seeds is hopeless, and the paired per-seed
delta used throughout is not a nicety but a requirement.
"""
import json

import numpy as np
import pandas as pd

RESULTS = "outputs/ablation/results.jsonl"
OUT = "outputs/ablation/noise_floor.json"
METRIC = "global_pcc_score"

# The three reproduction runs of the unmodified published script, random split,
# --seed 42, logged in outputs/LOG.md Entry 1.
UNSEEDED = {
    "global_pcc_score": [0.7242, 0.7190, 0.7195],
    "mean_allele_pcc": [0.5300, 0.5259, 0.5323],
    "rmse_hours": [10.89, 10.76, 10.98],
}


def stats(vals, prefix=""):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    return {
        f"{prefix}n": int(len(v)),
        f"{prefix}mean": float(v.mean()),
        f"{prefix}sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        f"{prefix}range": float(v.max() - v.min()),
        f"{prefix}min": float(v.min()),
        f"{prefix}max": float(v.max()),
    }


def main():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    df = pd.DataFrame(rows)

    out = {}

    # --- unseeded published script -------------------------------------
    u = {"n": len(UNSEEDED[METRIC])}
    for m, vals in UNSEEDED.items():
        s = stats(vals)
        u[f"{m}_mean"], u[f"{m}_sd"] = s["mean"], s["sd"]
        u[f"{m}_range"] = s["range"]
    out["unseeded"] = u

    # --- initialisation only (split fixed, torch seed varied) ----------
    init = df[(df.get("tag") == "noise_init")]
    if len(init):
        d = {"n": int(len(init))}
        for m in ["global_pcc_score", "mean_allele_pcc", "rmse_hours", "auc_1h"]:
            s = stats(init[m].to_numpy())
            d[f"{m}_mean"], d[f"{m}_sd"] = s["mean"], s["sd"]
            d[f"{m}_range"] = s["range"]
        d["torch_seeds"] = sorted(int(x) for x in init["torch_seed"].unique())
        out["init"] = d

    # --- split + init (seed varied) ------------------------------------
    base = df[(df["pep"] == "blosum_pep") & (df["hla"] == "blosum_hla")
              & (df["split"] == "random") & (df.get("tag") == "")
              & (df["epochs"] == 25) & (df["hidden_dim"] == 60)]
    if len(base):
        d = {"n": int(len(base))}
        for m in ["global_pcc_score", "mean_allele_pcc", "rmse_hours", "auc_1h"]:
            s = stats(base[m].to_numpy())
            d[f"{m}_mean"], d[f"{m}_sd"] = s["mean"], s["sd"]
            d[f"{m}_range"] = s["range"]
        d["seeds"] = sorted(int(x) for x in base["seed"].unique())
        out["split"] = d

    # --- the single threshold used across the project ------------------
    # Take the larger of (init SD, unseeded SD) as the floor: it is the spread
    # that is NOT attributable to the data, so a difference below it cannot be
    # a property of the features.
    cands = [out.get("init", {}).get("global_pcc_score_sd"),
             out["unseeded"]["global_pcc_score_sd"]]
    cands = [c for c in cands if c]
    out["global_pcc_score_sd"] = float(max(cands)) if cands else 0.005

    # Per-split seed spread of the baseline arm. This matters because the three
    # splits behave completely differently: a random split of 28k rows is very
    # stable, while leave-allele-out variance is dominated by WHICH 15 alleles
    # are held out -- which swamps any feature effect.
    per_split = {}
    for split in ["random", "cluster", "allele"]:
        b = df[(df["pep"] == "blosum_pep") & (df["hla"] == "blosum_hla")
               & (df["split"] == split) & (df.get("tag") == "")
               & (df["epochs"] == 25) & (df["hidden_dim"] == 60)]
        if not len(b):
            continue
        d = {"n": int(len(b))}
        for m in ["global_pcc_score", "mean_allele_pcc"]:
            s = stats(b[m].to_numpy())
            d[f"{m}_mean"], d[f"{m}_sd"] = s["mean"], s["sd"]
            d[f"{m}_range"] = s["range"]
        per_split[split] = d
    out["per_split_seed_spread"] = per_split

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[+] wrote {OUT}")
    for k in ["unseeded", "init", "split"]:
        if k in out:
            o = out[k]
            print(f"  {k:9s} n={o['n']:2d}  global_pcc {o['global_pcc_score_mean']:.4f} "
                  f"sd={o['global_pcc_score_sd']:.4f} range={o['global_pcc_score_range']:.4f}")
    print(f"  floor used for figures: sd={out['global_pcc_score_sd']:.4f}")
    print("\n  baseline seed spread per split (global PCC):")
    for split, d in out["per_split_seed_spread"].items():
        print(f"    {split:8s} n={d['n']} mean={d['global_pcc_score_mean']:.4f} "
              f"sd={d['global_pcc_score_sd']:.4f} range={d['global_pcc_score_range']:.4f}")
    rnd = out["per_split_seed_spread"].get("random", {}).get("global_pcc_score_sd")
    alle = out["per_split_seed_spread"].get("allele", {}).get("global_pcc_score_sd")
    if rnd and alle:
        print(f"\n  leave-allele-out seed SD is {alle/rnd:.0f}x the random-split seed SD.")
        print("  On random, init noise dominates and splits are near-identical; on")
        print("  leave-allele-out, WHICH alleles are held out dominates everything.")
        print("  Either way the arms must be compared paired within a seed.")


if __name__ == "__main__":
    main()
