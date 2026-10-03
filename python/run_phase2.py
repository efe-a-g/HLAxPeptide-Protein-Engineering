#!/usr/bin/env python3
"""
Phase 2: probes that decide WHY an arm won or lost, rather than just that it did.

Phase 1 showed (a) Boltz-HLA beats BLOSUM-HLA when test alleles are also in
training but collapses on unseen alleles, and (b) ESM-2 peptide features lose
heavily. Neither observation interprets itself:

  noise    Separate initialisation noise from split noise: fix the split
           (--seed 42) and vary --torch_seed. Establishes the floor properly.

  capacity Is the ESM-2 peptide loss a worse representation, or just an
           underfitted 5,760-dim input in a 60-unit head on a 25-epoch budget?
           Probe: more epochs, a wider head, and PCA down to BLOSUM-peptide
           width (180) so input widths match. If ESM recovers under any of
           these, the phase-1 loss was an optimisation artifact, not a verdict
           on the representation.

  pooling  Mean-pooled vs per-residue ESM-2, and a smaller ESM-2 (35M). For
           9-mers, mean pooling destroys the anchor-position information that
           peptide-MHC binding depends on, so this bounds how much of the loss
           is a pooling choice.

  identity Does Boltz-2 carry anything beyond allele identity? Compare
           Boltz-HLA against a 75-dim allele one-hot on every split, and
           against BLOSUM-HLA augmented with Boltz. Run on all splits because
           the whole point is the random-vs-allele contrast.
"""
import argparse
import itertools
import time

import run_ablation

PEP_B, PEP_E = "blosum_pep", "esm2_150m_pep_res"
HLA_B, HLA_Z, HLA_1 = "blosum_hla", "boltz_BF", "onehot_hla"
SEEDS = [42, 43, 44, 45, 46]


def launch(argv_list, results, label):
    print(f"\n{'='*78}\n[PHASE2] {label}  ({len(argv_list)} runs)\n{'='*78}", flush=True)
    t0 = time.time()
    for i, argv in enumerate(argv_list, 1):
        try:
            run_ablation.main(argv + ["--results", results, "--quiet"])
        except Exception as e:
            print(f"[!] FAILED {argv}: {e}", flush=True)
        el = time.time() - t0
        print(f"    [{i}/{len(argv_list)}] {el/60:.1f}m elapsed, "
              f"eta {(el/i)*(len(argv_list)-i)/60:.1f}m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+",
                    default=["noise", "capacity", "pooling"])
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    args = ap.parse_args()

    if "noise" in args.probes:
        # Split FIXED at seed 42; only weight init / shuffling varies.
        runs = [["--pep", PEP_B, "--hla", HLA_B, "--split_strategy", "random",
                 "--seed", "42", "--torch_seed", str(ts), "--tag", "noise_init"]
                for ts in [101, 102, 103, 104, 105, 106, 107, 108]]
        launch(runs, args.results, "noise: initialisation only (split fixed)")

    if "capacity" in args.probes:
        runs = []
        # More epochs, at the published head width.
        for pep, ep in itertools.product([PEP_B, PEP_E], [100]):
            for split, seed in itertools.product(["random", "cluster", "allele"], SEEDS):
                runs.append(["--pep", pep, "--hla", HLA_B,
                             "--split_strategy", split, "--seed", str(seed),
                             "--epochs", str(ep), "--tag", f"ep{ep}"])
        # Wider head, 25 epochs.
        for pep in [PEP_B, PEP_E]:
            for split, seed in itertools.product(["random", "cluster", "allele"], SEEDS):
                runs.append(["--pep", pep, "--hla", HLA_B,
                             "--split_strategy", split, "--seed", str(seed),
                             "--hidden_dim", "256", "--tag", "wide256"])
        # Dimension-matched: learned blocks projected to BLOSUM-peptide width.
        for pep in [PEP_E]:
            for split, seed in itertools.product(["random", "cluster", "allele"], SEEDS):
                runs.append(["--pep", pep, "--hla", HLA_B,
                             "--split_strategy", split, "--seed", str(seed),
                             "--pca_dim", "180", "--tag", "pca180"])
        launch(runs, args.results, "capacity: epochs / width / dimension-matched")

    if "pooling" in args.probes:
        runs = []
        for pep in ["esm2_150m_pep_mean", "esm2_35m_pep_res"]:
            for split, seed in itertools.product(["random", "cluster", "allele"], SEEDS):
                runs.append(["--pep", pep, "--hla", HLA_B,
                             "--split_strategy", split, "--seed", str(seed),
                             "--tag", "pooling"])
        launch(runs, args.results, "pooling: mean-pool vs per-residue, 35M vs 150M")

    if "augment" in args.probes:
        # AUGMENT rather than replace: keep the pseudosequence and add the
        # learned HLA block on top. This is the configuration that matters in
        # practice -- "does it add anything the pseudosequence lacks?" -- and
        # it is the one the brief explicitly allows alongside replacement.
        runs = []
        for hla in ["blosum_hla+boltz_BF", "blosum_hla+esm2_150m_hla_mean",
                    "blosum_hla+boltz_BF+esm2_150m_hla_mean"]:
            for split, seed in itertools.product(["random", "cluster", "allele"], SEEDS):
                runs.append(["--pep", PEP_B, "--hla", hla,
                             "--split_strategy", split, "--seed", str(seed),
                             "--tag", "augment"])
        launch(runs, args.results, "augment: pseudosequence + learned HLA block")

    if "identity" in args.probes:
        runs = []
        for hla in [HLA_1, "boltz_BFs"]:
            for split, seed in itertools.product(["random", "cluster", "allele"], SEEDS):
                runs.append(["--pep", PEP_B, "--hla", hla,
                             "--split_strategy", split, "--seed", str(seed),
                             "--tag", "identity"])
        launch(runs, args.results, "identity: Boltz vs allele one-hot")

    print("\n[+] phase 2 complete", flush=True)


if __name__ == "__main__":
    main()
