#!/usr/bin/env python3
"""
Re-run the feature ablation under the PROJECT-WIDE convention from baseline.py.

baseline.py (origin/baseline @ ed9d863) states that every model in the repo must
import `split_by_supertype` and `evaluate` from it and use them UNCHANGED, so
results are comparable across approaches. My earlier grid predates that commit
and does not comply: it used a naive random allele holdout at test_size 0.20
with no supertype stratification and no measurement floor. Those numbers are
internally consistent but not comparable to anything else in this repo.

This harness complies. It imports the split, the metric, the target transform,
the BLOSUM encoder, the model and the null baseline from baseline.py, and varies
only the input features.

Differences from my earlier grid, all inherited from baseline.py:
  * supertype-stratified leave-allele-out, split on unique PSEUDOSEQUENCE so a
    C67S construct cannot straddle train and test
  * test_size 0.25, and an allele is test-eligible only with >= 100 measurements
    (a Spearman over ~16 points is noise but would carry equal weight in the mean)
  * 90 epochs, no validation split, no early stopping
  * headline metric is mean per-allele Spearman

`peptide_only_null` is included as an arm because baseline.py argues the real
model must beat it: Rasmussen screened one peptide panel per motif group against
every allele in that group, so most test peptides were already seen paired with
other alleles, and a model that learnt nothing about alleles still scores well.
"""
import argparse
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline as B                      # the canonical split / metric / model
from run_ablation import build_side, fit_transform_blocks

DATA = "data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv"

ARMS = [
    ("blosum_pep", "blosum_hla"),                       # the published baseline
    ("blosum_pep", "boltz_BF"),                         # Boltz-2 replaces HLA side
    ("blosum_pep", "blosum_hla+boltz_BF"),              # Boltz-2 augments HLA side
    ("blosum_pep", "esm2_150m_hla_mean"),               # ESM-2 replaces HLA side
    ("blosum_pep", "onehot_hla"),                       # allele identity control
    ("esm2_150m_pep_res", "blosum_hla"),                # ESM-2 replaces peptide side
    ("esm2_150m_pep_res", "boltz_BF"),                  # both sides learned
]


def run_one(df, pep, hla, seed, args, results_path):
    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    # UNCHANGED from baseline.py -- this is the whole point of the exercise.
    train_pos, test_pos = B.split_by_supertype(
        df, test_size=args.test_size, min_measurements=args.min_measurements,
        seed=seed, verbose=False)

    fargs = SimpleNamespace(pca_dim=0, standardize_blosum=False)
    blocks = build_side(df, pep, fargs) + build_side(df, hla, fargs)
    # No validation split in this protocol, so fit scalers on train and pass the
    # train rows twice; only the train statistics are ever used.
    X_tr, _, X_te, dims = fit_transform_blocks(
        blocks, train_pos, train_pos[:1], test_pos, fargs)

    y_all = B.transform_target(df["thalf_hours"].to_numpy(), t0=args.t0)
    y_tr, y_te = y_all[train_pos], y_all[test_pos]

    targs = SimpleNamespace(device="cpu", hidden_dim=args.hidden_dim, lr=args.lr,
                            epochs=args.epochs, batch_size=args.batch_size)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        model, predict = B.train_model(X_tr, y_tr, targs)
        y_pred = predict(X_te)

    alleles = df["allele"].to_numpy()[test_pos]
    metrics, per_allele = B.evaluate(alleles, df["thalf_hours"].to_numpy()[test_pos],
                                     y_te, y_pred)

    rec = {"pep": pep, "hla": hla, "label": f"{pep} | {hla}", "seed": seed,
           "epochs": args.epochs, "hidden_dim": args.hidden_dim,
           "split": "supertype", "protocol": "baseline.py@ed9d863",
           "dims": dims, "n_features": int(X_tr.shape[1]),
           "n_train": int(len(train_pos)), "n_test": int(len(test_pos)),
           "runtime_s": round(time.time() - t0, 1), **metrics}
    with open(results_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

    os.makedirs("outputs/supertype/per_allele", exist_ok=True)
    per_allele.to_csv(f"outputs/supertype/per_allele/{pep}__{hla}__s{seed}.csv",
                      index=False)
    print(f"[+] {pep + ' | ' + hla:46s} s{seed} "
          f"| allele_scc {metrics['mean_allele_scc']:.4f} "
          f"| allele_pcc {metrics['mean_allele_pcc']:.4f} "
          f"| median_scc {metrics['median_allele_scc']:.4f} "
          f"| n={metrics['n_alleles_scored']:2d} | {rec['runtime_s']}s", flush=True)


def run_null(df, seed, args, results_path):
    """baseline.py's peptide-only null, scored through the same evaluate()."""
    t0 = time.time()
    train_pos, test_pos = B.split_by_supertype(
        df, test_size=args.test_size, min_measurements=args.min_measurements,
        seed=seed, verbose=False)
    y_all = B.transform_target(df["thalf_hours"].to_numpy(), t0=args.t0)
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]
    y_pred = B.peptide_only_null(train_df, test_df, y_all[train_pos])
    metrics, per_allele = B.evaluate(
        df["allele"].to_numpy()[test_pos], df["thalf_hours"].to_numpy()[test_pos],
        y_all[test_pos], y_pred)
    rec = {"pep": "peptide_mean_null", "hla": "none",
           "label": "peptide-only null", "seed": seed, "epochs": 0,
           "hidden_dim": 0, "split": "supertype",
           "protocol": "baseline.py@ed9d863", "dims": {}, "n_features": 0,
           "n_train": int(len(train_pos)), "n_test": int(len(test_pos)),
           "runtime_s": round(time.time() - t0, 1), **metrics}
    with open(results_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    per_allele.to_csv(
        f"outputs/supertype/per_allele/peptide_mean_null__none__s{seed}.csv",
        index=False)
    print(f"[+] {'peptide-only null':46s} s{seed} "
          f"| allele_scc {metrics['mean_allele_scc']:.4f} "
          f"| allele_pcc {metrics['mean_allele_pcc']:.4f} "
          f"| median_scc {metrics['median_allele_scc']:.4f} "
          f"| n={metrics['n_alleles_scored']:2d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=90)      # baseline.py default
    ap.add_argument("--hidden_dim", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--min_measurements", type=int, default=100)
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--results", default="outputs/supertype/results.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    os.makedirs("outputs/supertype/per_allele", exist_ok=True)
    df = pd.read_csv(DATA)

    missing = set(df["allele"]) - set(B.SUPERTYPE)
    if missing:
        raise SystemExit(f"[!] alleles missing from SUPERTYPE table: {sorted(missing)}")

    # Report the split once so its shape is on the record.
    B.split_by_supertype(df, test_size=args.test_size,
                         min_measurements=args.min_measurements,
                         seed=args.seeds[0], verbose=True)

    total = len(ARMS) * len(args.seeds) + len(args.seeds)
    print(f"[*] {len(ARMS)} arms + null x {len(args.seeds)} seeds = {total} runs "
          f"at {args.epochs} epochs\n", flush=True)

    t0 = time.time()
    done = 0
    for seed in args.seeds:
        run_null(df, seed, args, args.results)
        done += 1
        for pep, hla in ARMS:
            try:
                run_one(df, pep, hla, seed, args, args.results)
            except Exception as e:
                print(f"[!] FAILED {pep} | {hla} s{seed}: {e}", flush=True)
            done += 1
            el = time.time() - t0
            print(f"    [{done}/{total}] {el/60:.1f}m elapsed, "
                  f"eta {(el/done)*(total-done)/60:.1f}m", flush=True)
    print(f"\n[+] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
