#!/usr/bin/env python3
"""Verify the dataset and Boltz-2 embedding parquets match the documented facts."""
import numpy as np
import pandas as pd

DATA = "data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv"

df = pd.read_csv(DATA)
print(f"CSV rows={len(df)} cols={list(df.columns)}")
print(f"  unique alleles      : {df['allele'].nunique()}")
print(f"  unique peptides      : {df['peptide'].nunique()}")
print(f"  peptide lengths      : {sorted(df['peptide'].str.len().unique())}")
print(f"  hla_seq lengths      : {sorted(df['hla_seq'].str.len().unique())}")
print(f"  pseudoseq lengths    : {sorted(df['hla_pseudoseq'].str.len().unique())}")
print(f"  thalf_hours          : min={df['thalf_hours'].min():.4f} "
      f"max={df['thalf_hours'].max():.4f} median={df['thalf_hours'].median():.4f} "
      f"nan={df['thalf_hours'].isna().sum()}")
print(f"  thalf == 0 rows      : {(df['thalf_hours'] <= 0).sum()}")
print(f"  non-standard AA in peptide: "
      f"{df['peptide'].str.contains('[^ARNDCQEGHILKMFPSTWYV]').sum()}")
print(f"  duplicated (allele,peptide): {df.duplicated(['allele','peptide']).sum()}")
print(f"  alleles with suffix  : {sorted(a for a in df['allele'].unique() if '(' in a)}")

# one pseudoseq per allele?
g = df.groupby("allele")["hla_pseudoseq"].nunique()
print(f"  alleles w/ >1 pseudoseq: {(g > 1).sum()}")
g2 = df.groupby("hla_pseudoseq")["allele"].nunique()
print(f"  pseudoseqs shared by >1 allele: {(g2 > 1).sum()}")
print(f"  unique pseudoseqs: {df['hla_pseudoseq'].nunique()}")

for name in ["boltz_pockets", "boltz_full"]:
    print(f"\n--- embeddings/{name}.parquet ---")
    e = pd.read_parquet(f"embeddings/{name}.parquet")
    cols = list(e.columns)
    meta = [c for c in cols if not c.startswith("boltz_")]
    print(f"rows={len(e)} total_cols={len(cols)}")
    print(f"  meta cols: {meta}")
    for pre in ["boltz_B_", "boltz_F_", "boltz_s_"]:
        n = sum(c.startswith(pre) for c in cols)
        if n:
            print(f"  {pre}* : {n} dims")
    # per-allele check: how many distinct embedding vectors?
    bcols = [c for c in cols if c.startswith("boltz_")]
    sub = e[["allele"] + bcols]
    nun = sub.groupby("allele")[bcols[0]].nunique()
    print(f"  alleles with >1 distinct value in {bcols[0]}: {(nun > 1).sum()} / {len(nun)}")
    arr = e[bcols].to_numpy(dtype=np.float32)
    uniq_rows = np.unique(arr, axis=0).shape[0]
    print(f"  distinct embedding rows: {uniq_rows}  (alleles: {e['allele'].nunique()})")
    print(f"  value range: [{arr.min():.4f}, {arr.max():.4f}]  "
          f"nan={np.isnan(arr).sum()}  norm(mean)={np.linalg.norm(arr,axis=1).mean():.3f}")
    # row alignment with CSV
    key_e = e["allele"].astype(str) + "|" + e["peptide"].astype(str)
    key_d = df["allele"].astype(str) + "|" + df["peptide"].astype(str)
    print(f"  identical row order vs CSV: {bool((key_e.values == key_d.values).all())}")
