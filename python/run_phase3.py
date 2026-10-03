#!/usr/bin/env python3
"""
Phase 3, re-prioritised after the 100-epoch capacity probe.

What that probe showed, and why it changes the plan:

  At 100 epochs the BLOSUM baseline improves a lot (mean per-allele PCC
  0.536 -> 0.629 on the random split) while ESM-2 peptide features barely move
  (0.430 -> 0.439). So the ESM-2 peptide deficit is NOT underfitting -- the
  25-epoch budget was flattering to ESM-2, and the gap widens with proper
  training. That question is answered; the 256-unit probe testing the same
  hypothesis was stopped as redundant and slow.

  It also means the published 25-epoch protocol is UNDERTRAINED, and that
  every phase-1 conclusion was drawn at that undertrained setting. Longer
  training buys the baseline ~+0.09 per-allele PCC, roughly 3x what the best
  foundation-model feature buys it. So the urgent question is now:

      does the Boltz-2 / ESM-2 HLA-side gain SURVIVE at 100 epochs,
      or was it only compensating for an undertrained baseline?

Probes, in priority order so that whatever finishes first is the most useful:

  hla100   The critical one. HLA-side arms at 100 epochs, all splits.
  pca      Dimension-matched ESM-2 peptide (PCA to BLOSUM width, 180) --
           separates representation quality from first-layer parameter count.
  augment  Pseudosequence + learned block rather than replacement, at both
           25 and 100 epochs.
  pooling  Mean-pooled vs per-residue ESM-2, and 35M vs 150M.
"""
import argparse
import itertools
import time

import run_ablation

PEP_B, PEP_E = "blosum_pep", "esm2_150m_pep_res"
HLA_B, HLA_Z, HLA_E, HLA_1 = "blosum_hla", "boltz_BF", "esm2_150m_hla_mean", "onehot_hla"
SEEDS = [42, 43, 44, 45, 46]
SPLITS = ["random", "cluster", "allele"]


def launch(runs, results, label):
    print(f"\n{'='*78}\n[PHASE3] {label}  ({len(runs)} runs)\n{'='*78}", flush=True)
    t0 = time.time()
    for i, argv in enumerate(runs, 1):
        try:
            run_ablation.main(argv + ["--results", results, "--quiet"])
        except Exception as e:
            print(f"[!] FAILED {argv}: {e}", flush=True)
        el = time.time() - t0
        print(f"    [{i}/{len(runs)}] {el/60:.1f}m elapsed, "
              f"eta {(el/i)*(len(runs)-i)/60:.1f}m", flush=True)
    print(f"[PHASE3] done: {label} in {(time.time()-t0)/60:.1f}m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+",
                    default=["hla100", "pca", "augment", "pooling"])
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    args = ap.parse_args()

    if "hla100" in args.probes:
        # The baseline at 100 epochs already exists from the capacity probe,
        # so only the comparison arms are needed to form paired deltas.
        runs = [["--pep", PEP_B, "--hla", h, "--split_strategy", s,
                 "--seed", str(sd), "--epochs", "100", "--tag", "ep100"]
                for h, s, sd in itertools.product([HLA_Z, HLA_E, HLA_1], SPLITS, SEEDS)]
        launch(runs, args.results, "hla100: does the HLA-side gain survive 100 epochs?")

    if "pca" in args.probes:
        runs = [["--pep", PEP_E, "--hla", HLA_B, "--split_strategy", s,
                 "--seed", str(sd), "--pca_dim", "180", "--tag", "pca180"]
                for s, sd in itertools.product(SPLITS, SEEDS)]
        launch(runs, args.results, "pca: ESM-2 peptide matched to BLOSUM width")

    if "augment" in args.probes:
        runs = []
        for hla, ep in itertools.product(
                ["blosum_hla+boltz_BF", "blosum_hla+esm2_150m_hla_mean"],
                ["25", "100"]):
            for s, sd in itertools.product(SPLITS, SEEDS):
                tag = "augment" if ep == "25" else "ep100"
                runs.append(["--pep", PEP_B, "--hla", hla, "--split_strategy", s,
                             "--seed", str(sd), "--epochs", ep, "--tag", tag])
        launch(runs, args.results, "augment: pseudosequence + learned block")

    if "pooling" in args.probes:
        runs = [["--pep", p, "--hla", HLA_B, "--split_strategy", s,
                 "--seed", str(sd), "--tag", "pooling"]
                for p, s, sd in itertools.product(
                    ["esm2_150m_pep_mean", "esm2_35m_pep_res"], SPLITS, SEEDS)]
        launch(runs, args.results, "pooling: mean vs per-residue, 35M vs 150M")

    print("\n[+] phase 3 complete", flush=True)


if __name__ == "__main__":
    main()
