#!/usr/bin/env python3
"""
Audit the three split strategies for leakage.

Motivated by a real risk: under pandas 3.0, df['peptide'].unique() returns an
ArrowStringArray, and np.random.shuffle warns that shuffling such an object
"may contain duplicates after shuffling". train_baseline.py's 'cluster' split
relies on exactly that shuffle, so if it misbehaves, held-out peptides can leak
into train and the peptide-grouped split would be silently invalid.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import get_train_test_split

df = pd.read_csv("data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")

print(f"dtype of df['peptide'].unique(): {type(df['peptide'].unique())}")
u = df["peptide"].unique()
arr = np.asarray(u, dtype=object).copy()
np.random.seed(0); np.random.shuffle(u)
print(f"  after shuffle on raw unique(): {len(u)} entries, "
      f"{len(set(map(str, u)))} distinct  -> "
      f"{'OK' if len(set(map(str,u))) == len(u) else 'DUPLICATES INTRODUCED'}")
np.random.seed(0); np.random.shuffle(arr)
print(f"  after shuffle on object copy : {len(arr)} entries, "
      f"{len(set(arr))} distinct")

ok = True
for strategy in ["random", "allele", "cluster"]:
    for seed in [42, 43, 44, 45, 46]:
        tr, te = get_train_test_split(df, strategy=strategy, test_size=0.2,
                                      seed=seed)
        tr, te = np.asarray(tr), np.asarray(te)
        n_overlap_rows = len(set(tr.tolist()) & set(te.tolist()))
        covered = len(tr) + len(te)
        pep_tr = set(df["peptide"].values[tr])
        pep_te = set(df["peptide"].values[te])
        al_tr = set(df["allele"].values[tr])
        al_te = set(df["allele"].values[te])
        pep_leak = len(pep_tr & pep_te)
        al_leak = len(al_tr & al_te)
        flags = []
        if n_overlap_rows:
            flags.append(f"ROW-OVERLAP={n_overlap_rows}")
        if covered != len(df):
            flags.append(f"COVERAGE={covered}/{len(df)}")
        if strategy == "cluster" and pep_leak:
            flags.append(f"PEPTIDE-LEAK={pep_leak}")
        if strategy == "allele" and al_leak:
            flags.append(f"ALLELE-LEAK={al_leak}")
        if flags:
            ok = False
        print(f"  {strategy:8s} s{seed}: train={len(tr):6d} test={len(te):6d} "
              f"| shared peptides={pep_leak:5d} shared alleles={al_leak:3d} "
              f"| test alleles={len(al_te):3d} "
              f"{'  <-- ' + ', '.join(flags) if flags else 'clean'}")

print("\nVERDICT:", "all splits clean" if ok else "PROBLEMS FOUND (see flags)")
