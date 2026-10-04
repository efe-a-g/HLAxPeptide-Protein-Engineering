#!/usr/bin/env python3
"""
Driver for loocv_distance.py: build the shared distance tables once, then fan the 75 folds out.

Running the fold script six times in parallel made each worker rebuild the 5,633x5,633 peptide
Hamming table and the per-allele neighbour tables from scratch, which costs more than the
training it was meant to overlap. Here the tables are built once in the parent and inherited by
the pool, so the workers do nothing but train.
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from baseline import AA_TO_IDX, evaluate, transform_target
from loocv_distance import hamming_table, neighbour_tables, nn_features, train_net

G = {}


def init(payload):
    G.update(payload)
    torch.set_num_threads(1)


def fold(a):
    t0 = time.time()
    allele_ids, s, X_base = G["allele_ids"], G["s"], G["X_base"]
    te = np.flatnonzero(allele_ids == a)
    tr = np.flatnonzero(allele_ids != a)
    args = (allele_ids, G["pep_ids"], G["hla_d"], G["PEPMIN"], G["SUMS"], G["CNT"])
    f_te = nn_features(te, a, *args)
    f_tr = nn_features(tr, a, *args)
    mu, sd = f_tr.mean(0), f_tr.std(0)
    sd[sd < 1e-8] = 1.0
    Xc_tr = np.hstack([X_base[tr], (f_tr - mu) / sd]).astype(np.float32)
    Xc_te = np.hstack([X_base[te], (f_te - mu) / sd]).astype(np.float32)
    ep, seed = G["epochs"], G["seed"]
    pred = {"nn": f_te[:, 0].astype(np.float64),
            "baseline": train_net(X_base[tr], s[tr], X_base[te], ep, seed),
            "combined": train_net(Xc_tr, s[tr], Xc_te, ep, seed)}
    al_te = G["allele_names"][te]
    row = {"allele": G["alleles"][a], "n": len(te),
           "mean_hamming": float(f_te[:, 1].mean()),
           "mean_hla_hamming": float(f_te[:, 2].mean()),
           "mean_peptide_hamming": float(f_te[:, 3].mean())}
    for k, v in pred.items():
        row[f"{k}_spearman"] = evaluate(al_te, None, s[te], v)[0]["mean_allele_scc"]
    print(f"{G['alleles'][a]:22s} n {len(te):5d} dist {row['mean_hamming']:6.2f} | "
          f"nn {row['nn_spearman']:+.3f} base {row['baseline_spearman']:+.3f} "
          f"comb {row['combined_spearman']:+.3f}  [{time.time() - t0:.0f}s]", flush=True)
    return row, {k: v for k, v in pred.items()}, te


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output_dir", default="outputs/loocv_distance")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(args.data_path).reset_index(drop=True)
    s = transform_target(df["thalf_hours"].values)
    alleles = np.sort(df["allele"].unique())
    allele_ids = pd.Categorical(df["allele"], categories=alleles).codes.astype(np.int32)
    peps = np.sort(df["peptide"].unique())
    pep_ids = pd.Categorical(df["peptide"], categories=peps).codes.astype(np.int32)
    one = df.drop_duplicates("allele").set_index("allele").loc[alleles]
    hla_d = hamming_table(one["hla_seq"].to_numpy())
    pep_d = hamming_table(peps).astype(np.int16)
    PEPMIN, SUMS, CNT = neighbour_tables(df, s, pep_d, hla_d, allele_ids, pep_ids, len(alleles))
    del pep_d
    idx = np.array([[AA_TO_IDX[c] for c in pp + hh]
                    for pp, hh in zip(df["peptide"], df["hla_pseudoseq"])])
    X_base = (np.eye(20, dtype=np.float32)[idx] * 0.85 + 0.05).reshape(len(df), -1)
    print(f"[*] tables built in {time.time() - t0:.0f}s | {len(alleles)} alleles, "
          f"{len(peps):,} peptides, {len(df):,} rows | {args.epochs} epochs/model\n", flush=True)

    payload = dict(allele_ids=allele_ids, pep_ids=pep_ids, hla_d=hla_d, PEPMIN=PEPMIN,
                   SUMS=SUMS, CNT=CNT, X_base=X_base, s=s, alleles=alleles,
                   allele_names=df["allele"].to_numpy(), epochs=args.epochs, seed=args.seed)
    csv_path = os.path.join(args.output_dir, "per_allele_all.csv")
    out = []
    with Pool(args.workers, initializer=init, initargs=(payload,)) as pool:
        for item in pool.imap_unordered(fold, range(len(alleles)), chunksize=1):
            out.append(item)
            # Rewrite after every fold so an interrupted run still leaves usable rows.
            pd.DataFrame([r for r, _, _ in out]).to_csv(csv_path, index=False)

    rows = [r for r, _, _ in out]
    preds = {}
    for r, pr, te in out:
        for k, v in pr.items():
            preds[f"{r['allele']}|{k}"] = v
        preds[f"{r['allele']}|rows"] = te
    np.savez(os.path.join(args.output_dir, "preds_all.npz"), s=s,
             allele=df["allele"].to_numpy(), **preds)
    print(f"\n[*] {len(rows)} folds in {time.time() - t0:.0f}s -> "
          f"{args.output_dir}/per_allele_all.csv")


if __name__ == "__main__":
    main()
