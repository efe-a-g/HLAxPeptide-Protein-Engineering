#!/usr/bin/env python3
"""
Drive the ablation grid in one process (module/parquet/ESM loads are reused).

Grid rationale -- the question is not "foundation model vs baseline" but which
*side* of the model benefits, and whether any benefit survives on alleles never
seen in training:

  main      the 2x2 the brief asks for: {BLOSUM, ESM} peptide x {BLOSUM, Boltz} HLA
  controls  allele one-hot HLA  -- if Boltz matches it, Boltz is contributing
                                   allele *identity*, not structure
            ESM-2 HLA           -- a sequence-derived foundation-model HLA
                                   representation that CAN extrapolate to
                                   unseen alleles, unlike per-allele vectors
            peptide-only / HLA-only -- how much signal each side carries alone
  matched   --pca_dim 180: learned blocks projected to BLOSUM-peptide width, to
                           separate representation quality from parameter count
"""
import argparse
import itertools
import sys
import time

import run_ablation

PEP_B, PEP_E = "blosum_pep", "esm2_150m_pep_res"
HLA_B, HLA_Z, HLA_E, HLA_1 = "blosum_hla", "boltz_BF", "esm2_150m_hla_mean", "onehot_hla"

GROUPS = {
    "main": [
        (PEP_B, HLA_B),   # = published baseline
        (PEP_B, HLA_Z),   # Boltz-2 replaces the HLA side
        (PEP_E, HLA_B),   # ESM-2 replaces the peptide side
        (PEP_E, HLA_Z),   # both sides from foundation models
    ],
    "controls": [
        (PEP_B, HLA_1),   # allele identity control
        (PEP_B, HLA_E),   # ESM-2 HLA instead of Boltz-2
        (PEP_E, HLA_E),
        (PEP_B, "none"),  # peptide only
        ("none", HLA_B),  # HLA only
    ],
    "augment": [
        (PEP_B, "boltz_BFs"),
        (PEP_B, "boltz_s"),
        (PEP_E, HLA_1),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["main", "controls"])
    ap.add_argument("--splits", nargs="+",
                    default=["random", "cluster", "allele"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    args = ap.parse_args()

    configs = []
    for g in args.groups:
        configs.extend(GROUPS[g])
    seen, uniq = set(), []
    for c in configs:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    total = len(uniq) * len(args.splits) * len(args.seeds)
    print(f"[*] {len(uniq)} configs x {len(args.splits)} splits x "
          f"{len(args.seeds)} seeds = {total} runs", flush=True)

    t0 = time.time()
    done = 0
    for (pep, hla), split, seed in itertools.product(
            uniq, args.splits, args.seeds):
        argv = ["--pep", pep, "--hla", hla, "--split_strategy", split,
                "--seed", str(seed), "--epochs", str(args.epochs),
                "--results", args.results, "--quiet"]
        if args.pca_dim:
            argv += ["--pca_dim", str(args.pca_dim)]
        if args.tag:
            argv += ["--tag", args.tag]
        try:
            run_ablation.main(argv)
        except Exception as e:  # keep the grid going; record the gap
            print(f"[!] FAILED {pep} + {hla} {split} s{seed}: {e}", flush=True)
        done += 1
        el = time.time() - t0
        print(f"    [{done}/{total}] elapsed {el/60:.1f}m, "
              f"eta {(el/done)*(total-done)/60:.1f}m", flush=True)

    print(f"[+] grid complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
