#!/usr/bin/env python3
"""
Ablation harness: which *part* of a foundation-model pipeline helps, if any?

Design rules (so the comparison stays honest):

  * Model, loss, target transform, metrics and split logic are IMPORTED from
    train_baseline.py -- never reimplemented. Every arm sees an identical
    60-unit sigmoid MLP, MSE on s = 2^(-t0/thalf), 25 epochs, Adam lr 1e-3,
    batch 128, best-validation checkpoint.
  * torch IS seeded here. train_baseline.py seeds only numpy, so its --seed
    fixes the split but not weight init; measured run-to-run spread at fixed
    seed is ~0.005 global PCC, which is the noise floor for all comparisons.
  * Learned blocks (ESM-2, Boltz-2) are z-scored with statistics fit on the
    TRAIN subset only. Boltz features reach magnitude ~1000 against BLOSUM's
    ~[-1,3]; fed raw into a sigmoid MLP they saturate it and nothing trains.
    BLOSUM blocks are left raw so the baseline arm matches train_baseline.py
    exactly (--standardize_blosum overrides this as a sanity check).
  * Optional --pca_dim projects each learned block down so a high-dimensional
    arm can be compared against BLOSUM at matched input width, separating
    "better representation" from "more first-layer parameters".

The peptide side and the HLA side are chosen independently, because the Boltz-2
embeddings are per-allele (75 distinct vectors for 28,166 rows) and carry no
peptide information at all.
"""
import argparse
import json
import os
import sys
import time
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import (  # noqa: E402  -- identical protocol, imported not copied
    BLOSUM50_MATRIX,
    AA_TO_IDX,
    calculate_metrics,
    get_train_test_split,
    inverse_transform_target,
    train_pytorch_model,
    transform_target,
)

DATA = "data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv"
CACHE = "cache"

# Blocks that are learned representations -> z-scored (train-fit).
LEARNED_PREFIX = ("esm", "boltz")


# ---------------------------------------------------------------------------
# Feature block construction. Each returns (N, D) float32 aligned to df rows.
# ---------------------------------------------------------------------------
def blosum_block(seqs):
    L = len(seqs[0])
    idx = np.zeros((len(seqs), L), dtype=np.int32)
    for i, s in enumerate(seqs):
        for j, aa in enumerate(s):
            idx[i, j] = AA_TO_IDX.get(aa, 0)
    return BLOSUM50_MATRIX[idx].reshape(len(seqs), -1).astype(np.float32)


def onehot_block(values):
    cats = sorted(set(values))
    pos = {c: i for i, c in enumerate(cats)}
    out = np.zeros((len(values), len(cats)), dtype=np.float32)
    for i, v in enumerate(values):
        out[i, pos[v]] = 1.0
    return out


@lru_cache(maxsize=None)
def _esm_table(model_key, kind, pooling):
    stem = f"{CACHE}/{model_key}_{kind}"
    with open(stem + "_index.json") as f:
        index = json.load(f)
    z = np.load(stem + ".npz")
    if pooling not in z:
        raise SystemExit(f"[!] pooling '{pooling}' not stored in {stem}.npz")
    table = z[pooling]
    return index, table.reshape(table.shape[0], -1).astype(np.float32)


def esm_block(df, model_key, kind, pooling):
    """Look up cached ESM-2 embeddings and expand to one row per dataset row."""
    index, table = _esm_table(model_key, kind, pooling)
    col = "peptide" if kind == "pep" else "hla_seq"
    rows = np.fromiter((index[s] for s in df[col].values), dtype=np.int64,
                       count=len(df))
    return table[rows]


@lru_cache(maxsize=None)
def boltz_block(which):
    e = pd.read_parquet("embeddings/boltz_pockets.parquet")
    groups = {"B": ["boltz_B_"], "F": ["boltz_F_"], "s": ["boltz_s_"],
              "BF": ["boltz_B_", "boltz_F_"],
              "BFs": ["boltz_B_", "boltz_F_", "boltz_s_"]}[which]
    cols = [c for c in e.columns if any(c.startswith(g) for g in groups)]
    cols.sort(key=lambda c: (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])))
    return e[cols].to_numpy(dtype=np.float32)


def build_side(df, spec, args):
    """
    spec -> list of (name, features (N,D), is_learned).

    A '+'-joined spec concatenates blocks, e.g. 'blosum_hla+boltz_BF' keeps the
    pseudosequence and *augments* it with the Boltz pockets rather than
    replacing it. Each sub-block keeps its own standardization flag, since
    mixing raw BLOSUM with z-scored Boltz in one block would be wrong.
    """
    if "+" in spec:
        out = []
        for part in spec.split("+"):
            out.extend(build_side(df, part, args))
        return out
    X, learned = _build_one(df, spec, args)
    return [] if X is None else [(spec, X, learned)]


def _build_one(df, spec, args):
    """spec -> (features (N,D) or None, is_learned)."""
    if spec == "none":
        return None, False
    if spec == "blosum_pep":
        return blosum_block(df["peptide"].values), False
    if spec == "blosum_hla":
        return blosum_block(df["hla_pseudoseq"].values), False
    if spec == "onehot_hla":
        return onehot_block(df["allele"].values), False
    if spec.startswith("boltz_"):
        return boltz_block(spec.split("_", 1)[1]), True
    if spec.startswith("esm"):
        # esm2_150m_pep_res / esm2_35m_hla_mean / ...
        parts = spec.split("_")
        model_key = "_".join(parts[:2])
        kind, pooling = parts[2], parts[3]
        return esm_block(df, model_key, kind, pooling), True
    raise SystemExit(f"[!] unknown feature spec: {spec}")


# ---------------------------------------------------------------------------
def fit_transform_blocks(blocks, tr, va, te, args):
    """Z-score (and optionally PCA) each block using TRAIN rows only."""
    out_tr, out_va, out_te, dims = [], [], [], {}
    for name, X, learned in blocks:
        if X is None:
            continue
        standardize = learned or args.standardize_blosum
        A, B, C = X[tr], X[va], X[te]
        if standardize:
            mu = A.mean(axis=0, keepdims=True)
            sd = A.std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            A, B, C = (A - mu) / sd, (B - mu) / sd, (C - mu) / sd
        if learned and args.pca_dim and A.shape[1] > args.pca_dim:
            from sklearn.decomposition import PCA
            p = PCA(n_components=args.pca_dim, random_state=0).fit(A)
            A, B, C = p.transform(A), p.transform(B), p.transform(C)
        dims[name] = int(A.shape[1])
        out_tr.append(np.ascontiguousarray(A, dtype=np.float32))
        out_va.append(np.ascontiguousarray(B, dtype=np.float32))
        out_te.append(np.ascontiguousarray(C, dtype=np.float32))
    if not out_tr:
        raise SystemExit("[!] no features selected")
    return (np.concatenate(out_tr, axis=1), np.concatenate(out_va, axis=1),
            np.concatenate(out_te, axis=1), dims)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pep", default="blosum_pep")
    ap.add_argument("--hla", default="blosum_hla")
    ap.add_argument("--split_strategy", default="random",
                    choices=["random", "allele", "cluster"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--torch_seed", type=int, default=-1,
                    help="Seed for weight init / batch shuffling. Default -1 "
                         "means 'same as --seed'. Setting it independently "
                         "isolates initialisation noise from split noise.")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden_dim", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--target_transform", default="netmhc")
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--standardize_blosum", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    # Seed everything, including torch -- see module docstring. --seed drives
    # the split; --torch_seed drives weight init and shuffling, defaulting to
    # --seed so one number reproduces a run.
    tseed = args.seed if args.torch_seed < 0 else args.torch_seed
    np.random.seed(args.seed)
    torch.manual_seed(tseed)
    torch.use_deterministic_algorithms(False)

    df = pd.read_csv(DATA)

    # Split, then carve validation out of train (same procedure as baseline).
    train_idx, test_idx = get_train_test_split(
        df, strategy=args.split_strategy, test_size=args.test_size, seed=args.seed)
    from sklearn.model_selection import train_test_split
    sub_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_size, random_state=args.seed, shuffle=True)

    blocks = build_side(df, args.pep, args) + build_side(df, args.hla, args)
    X_tr, X_va, X_te, dims = fit_transform_blocks(
        blocks, sub_idx, val_idx, test_idx, args)

    y_tr = transform_target(df["thalf_hours"].values[sub_idx],
                            mode=args.target_transform, t0=args.t0)
    y_va = transform_target(df["thalf_hours"].values[val_idx],
                            mode=args.target_transform, t0=args.t0)
    y_te_score = transform_target(df["thalf_hours"].values[test_idx],
                                  mode=args.target_transform, t0=args.t0)
    y_te_thalf = df["thalf_hours"].values[test_idx]

    label = f"{args.pep} + {args.hla}"
    if not args.quiet:
        print(f"[*] {label} | split={args.split_strategy} seed={args.seed}")
        print(f"[*] dims={dims} total={X_tr.shape[1]} "
              f"train={len(sub_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_args = SimpleNamespace(
        device=args.device, hidden_dim=args.hidden_dim,
        target_transform=args.target_transform, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size)

    import contextlib, io
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if args.quiet else contextlib.nullcontext()
    with ctx:
        _, predict_fn = train_pytorch_model(X_tr, y_tr, X_va, y_va, train_args)
    y_pred_score = predict_fn(X_te)
    y_pred_thalf = inverse_transform_target(
        y_pred_score, mode=args.target_transform, t0=args.t0)

    metrics, allele_metrics = calculate_metrics(
        y_true_thalf=y_te_thalf, y_pred_thalf=y_pred_thalf,
        y_true_score=y_te_score, y_pred_score=y_pred_score,
        alleles=df["allele"].values[test_idx])

    record = {
        "pep": args.pep, "hla": args.hla, "label": label,
        "split": args.split_strategy, "seed": args.seed, "torch_seed": tseed,
        "epochs": args.epochs, "hidden_dim": args.hidden_dim,
        "pca_dim": args.pca_dim, "tag": args.tag,
        "dims": dims, "n_features": int(X_tr.shape[1]),
        "n_train": len(sub_idx), "n_test": len(test_idx),
        "runtime_s": round(time.time() - t_start, 1),
        **metrics,
    }
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Per-allele detail, needed for the per-allele figures in the report.
    pa_dir = "outputs/ablation/per_allele"
    os.makedirs(pa_dir, exist_ok=True)
    slug = (f"{args.pep}__{args.hla}__{args.split_strategy}__s{args.seed}"
            f"{'__' + args.tag if args.tag else ''}")
    allele_metrics.to_csv(f"{pa_dir}/{slug}.csv", index=False)

    print(f"[+] {label:46s} {args.split_strategy:8s} s{args.seed} "
          f"| allele_pcc {metrics['mean_allele_pcc']:.4f} "
          f"| global_pcc {metrics['global_pcc_score']:.4f} "
          f"| auc1h {metrics['auc_1h']:.4f} "
          f"| rmse {metrics['rmse_hours']:.2f} "
          f"| {record['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
