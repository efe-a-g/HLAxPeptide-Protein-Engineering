#!/usr/bin/env python3
"""
Compute ESM-2 embeddings for the peptide and HLA sides of the Rasmussen dataset.

Everything is cached to disk keyed by (model, kind, sequence) so nothing is ever
recomputed. Two pooling modes are stored:

  'res'  per-residue representations, shape (L, H) -- the positional analogue of
         BLOSUM encoding. For 9-mers this flattens to 9*H dims.
  'mean' mean over residues, shape (H,) -- used for the 182-aa HLA domain.

CPU only (no CUDA on this machine). 5,633 unique peptides and 75 unique HLA
sequences, so this is a minutes-scale job.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

DATA = "data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv"
CACHE = "cache"

MODELS = {
    "esm2_8m": "facebook/esm2_t6_8M_UR50D",
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
}


def embed(seqs, model_name, batch_size=64):
    """Return (per_residue list of (L,H) float32, mean array (N,H) float32)."""
    from transformers import AutoTokenizer, EsmModel

    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 8)

    res_out, mean_out = [], []
    t0 = time.time()
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            h = model(**enc).last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"]
        for j, s in enumerate(batch):
            # ESM-2 prepends <cls> and appends <eos>; residues are 1..len(s)
            r = h[j, 1 : 1 + len(s)].to(torch.float32).numpy()
            assert r.shape[0] == len(s), (r.shape, len(s))
            res_out.append(r)
            mean_out.append(r.mean(axis=0))
        if i % (batch_size * 10) == 0:
            done = min(i + batch_size, len(seqs))
            print(f"    {done}/{len(seqs)}  ({time.time() - t0:.1f}s)", flush=True)
    return res_out, np.stack(mean_out).astype(np.float32)


def run(kind, seqs, model_key, batch_size):
    """Embed a deduplicated sequence list, writing npz + index json to cache/."""
    os.makedirs(CACHE, exist_ok=True)
    stem = f"{CACHE}/{model_key}_{kind}"
    if os.path.isfile(stem + ".npz") and os.path.isfile(stem + "_index.json"):
        print(f"[=] cached: {stem}.npz")
        return
    print(f"[*] {kind}: {len(seqs)} unique sequences, model {MODELS[model_key]}")
    res, mean = embed(seqs, MODELS[model_key], batch_size=batch_size)
    lengths = {len(r) for r in res}
    payload = {"mean": mean}
    if len(lengths) == 1:
        # Uniform length (peptides): store a dense (N, L, H) array
        payload["res"] = np.stack(res).astype(np.float32)
    np.savez_compressed(stem + ".npz", **payload)
    with open(stem + "_index.json", "w") as f:
        json.dump({s: i for i, s in enumerate(seqs)}, f)
    shapes = {k: list(v.shape) for k, v in payload.items()}
    print(f"[+] wrote {stem}.npz  {shapes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["esm2_35m", "esm2_150m"],
                    choices=list(MODELS))
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    peptides = sorted(df["peptide"].unique())
    hla_seqs = sorted(df["hla_seq"].unique())
    print(f"[*] {len(peptides)} unique peptides (len "
          f"{sorted({len(p) for p in peptides})}), "
          f"{len(hla_seqs)} unique HLA domains (len "
          f"{sorted({len(h) for h in hla_seqs})})")

    for mk in args.models:
        run("pep", peptides, mk, args.batch_size)
        run("hla", hla_seqs, mk, max(8, args.batch_size // 8))


if __name__ == "__main__":
    main()
