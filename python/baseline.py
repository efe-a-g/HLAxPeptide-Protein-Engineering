#!/usr/bin/env python3
"""
NetMHCstabpan baseline for peptide-HLA class I complex stability prediction.

Reproduces the ANN described in Rasmussen et al. (2016), "Pan-Specific Prediction of
Peptide-MHC Class I Complex Stability, a Correlate of T Cell Immunogenicity",
J Immunol 197(4):1517-1524.

PROJECT-WIDE CONVENTION -- this applies to every model in this repo, not just the
baseline. Later models (foundation-model embeddings, fine-tuned heads, etc.) should
import `split_by_supertype` and `evaluate` from here and use them unchanged, so results
are comparable across approaches.

  Split   Hold out whole alleles, stratified by HLA supertype so that every supertype
          appears in both train and test. This measures generalisation to a *new allele
          within a known motif family* -- the personalised-immunotherapy case -- rather
          than extrapolation to an unseen motif family, which this dataset (75 alleles)
          is too small to support.

  Metric  Mean per-allele Spearman rank correlation over the held-out alleles. We care
          about ranking candidate peptides against a fixed allele, not about absolute
          half-life, so rank correlation measures the thing we actually want and is
          immune to the saturation at both ends of the rescaled target.

Supertypes come from Sidney et al. (2008), BMC Immunol 9:1, Additional file 1 -- chosen
over Rasmussen's own Table 1 because the two tie on variance explained (ICC lift +0.117
each over a matched-k null) and Sidney is *external* to this dataset. Rasmussen's
grouping was used to design the peptide panels, so scoring it against this data is
partly circular.

Features: 9-mer peptide + 34-aa HLA pseudosequence = 43 positions x 20 = 860 dims.
Target:   s = 2^(-t0 / t_half), the paper's rescaling with a global t0 of 1 hour.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Amino acid alphabet and BLOSUM50
# ---------------------------------------------------------------------------
AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Standard BLOSUM50, divided by 5.0 as in Rasmussen et al. 2016.
BLOSUM50_RAW = [
    [ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0],  # A
    [-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3],  # R
    [-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4],  # D
    [-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1],  # C
    [-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3],  # Q
    [-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3],  # E
    [ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4],  # G
    [-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4],  # H
    [-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4],  # I
    [-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1],  # L
    [-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3],  # K
    [-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1],  # M
    [-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1],  # F
    [-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3],  # P
    [ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0],  # T
    [-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 3,-3],  # W
    [-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 3, 8,-1],  # Y
    [ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5],  # V
]
BLOSUM50_MATRIX = np.array(BLOSUM50_RAW, dtype=np.float32) / 5.0

# ---------------------------------------------------------------------------
# HLA supertypes
# ---------------------------------------------------------------------------
# Sidney et al. (2008) BMC Immunol 9:1, Additional file 1 (12865_2007_146_MOESM1_ESM.xls),
# which assigns supertypes to 945 HLA-A/B alleles. All 75 alleles in this dataset are
# covered with no gaps. The six marked "Unclassified" are Sidney's own label, not a
# fallback applied here. "A01 A03" / "A01 A24" are Sidney's dual-supertype assignments.
SUPERTYPE = {
    "HLA-A*01:01": "A01",          "HLA-A*02:01": "A02",
    "HLA-A*02:03": "A02",          "HLA-A*02:05": "A02",
    "HLA-A*02:10": "Unclassified", "HLA-A*02:11": "A02",
    "HLA-A*02:12": "A02",          "HLA-A*02:16": "A02",
    "HLA-A*02:19": "A02",          "HLA-A*02:50": "A02",
    "HLA-A*03:01": "A03",          "HLA-A*11:01": "A03",
    "HLA-A*23:01": "A24",          "HLA-A*24:02": "A24",
    "HLA-A*24:03": "A24",          "HLA-A*24:07": "Unclassified",
    "HLA-A*24:19": "Unclassified", "HLA-A*25:01": "A01",
    "HLA-A*26:01": "A01",          "HLA-A*26:02": "A01",
    "HLA-A*26:03": "A01",          "HLA-A*29:02": "A01 A24",
    "HLA-A*30:01": "A01 A03",      "HLA-A*30:02": "A01",
    "HLA-A*31:01": "A03",          "HLA-A*32:01": "A01",
    "HLA-A*32:07": "A01",          "HLA-A*33:03": "A03",
    "HLA-A*43:01": "Unclassified", "HLA-A*66:01": "A03",
    "HLA-A*68:01": "A03",          "HLA-A*68:02": "A02",
    "HLA-A*68:23": "A03",          "HLA-A*69:01": "A02",
    "HLA-A*74:01": "A03",          "HLA-A*80:01": "A01",
    "HLA-B*07:02": "B07",          "HLA-B*08:01": "B08",
    "HLA-B*08:03": "B08",          "HLA-B*13:02": "Unclassified",
    "HLA-B*14:01(C67S)": "B27",    "HLA-B*14:02(C67S)": "B27",
    "HLA-B*15:01": "B62",          "HLA-B*15:02": "B62",
    "HLA-B*15:10": "B27",          "HLA-B*15:17": "B58",
    "HLA-B*18:01": "B44",          "HLA-B*27:02": "B27",
    "HLA-B*27:03": "B27",          "HLA-B*27:05": "B27",
    "HLA-B*27:20": "B27",          "HLA-B*35:01": "B07",
    "HLA-B*35:03": "B07",          "HLA-B*35:08": "B07",
    "HLA-B*39:01": "B27",          "HLA-B*39:02": "B27",
    "HLA-B*39:06(C67S)": "B27",    "HLA-B*39:10": "B07",
    "HLA-B*40:01": "B44",          "HLA-B*40:02": "B44",
    "HLA-B*41:01": "B44",          "HLA-B*42:01": "B07",
    "HLA-B*42:02": "Unclassified", "HLA-B*44:05": "B44",
    "HLA-B*45:01": "B44",          "HLA-B*46:01": "B62",
    "HLA-B*51:01": "B07",          "HLA-B*54:01": "B07",
    "HLA-B*55:01": "B07",          "HLA-B*56:01": "B07",
    "HLA-B*57:01": "B58",          "HLA-B*57:02": "B58",
    "HLA-B*57:03": "B58",          "HLA-B*58:01": "B58",
    "HLA-B*81:01": "B07",
}


# ---------------------------------------------------------------------------
# Encoding and target transform
# ---------------------------------------------------------------------------
def encode_sequences(peptides, pseudoseqs, encoding="blosum50"):
    """Encode 9-mer peptide + 34-aa pseudosequence into (N, 860) float32 features."""
    n_samples = len(peptides)
    combined = [p + h for p, h in zip(peptides, pseudoseqs)]

    indices = np.zeros((n_samples, 43), dtype=np.int32)
    for i, seq in enumerate(combined):
        for j, aa in enumerate(seq):
            indices[i, j] = AA_TO_IDX.get(aa, 0)

    if encoding == "blosum50":
        features = BLOSUM50_MATRIX[indices].reshape(n_samples, -1)
    elif encoding == "sparse":
        # NetMHCpan sparse encoding: 0.9 for the matching residue, 0.05 for the other 19.
        oh = np.eye(20, dtype=np.float32)[indices].reshape(n_samples, -1)
        features = oh * 0.85 + 0.05
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return features.astype(np.float32)


def transform_target(thalf_hours, t0=1.0):
    """Rescale half-life to s = 2^(-t0/th), bounded in [0, 1]. th == 0 maps to s == 0."""
    th = np.asarray(thalf_hours, dtype=np.float32)
    s = np.zeros_like(th)
    mask = th > 0
    s[mask] = 2.0 ** (-t0 / th[mask])
    return s


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_by_supertype(df, test_size=0.25, min_measurements=100, seed=42, verbose=True):
    """
    Supertype-stratified leave-allele-out split. Returns (train_pos, test_pos), both
    positional indices into `df`.

    Split units are unique pseudosequences rather than allele names: HLA-B*14:01(C67S)
    and HLA-B*14:02(C67S) have identical 34-mers and so are the same input to the model.
    Splitting them apart would place a model-identical allele in both halves.

    An allele is test-eligible only if it has at least `min_measurements` rows -- a
    Spearman over 16 points is noise, and it would carry equal weight in the mean.
    There is a clean gap in this dataset: 68 alleles have >= 220 rows, seven have <= 32.

    Alleles that are not test-eligible, those Sidney marks Unclassified, and any
    supertype left with fewer than two eligible units (which covers Sidney's dual
    singletons A01 A03 and A01 A24) all go to train.
    """
    counts = df["allele"].value_counts()
    unit_of = dict(zip(df["allele"], df["hla_pseudoseq"]))

    # Group alleles into split units, and check each unit has one supertype.
    units = {}
    for allele, unit in unit_of.items():
        units.setdefault(unit, []).append(allele)
    for unit, alleles in units.items():
        sts = {SUPERTYPE[a] for a in alleles}
        if len(sts) > 1:
            raise ValueError(f"Pseudosequence shared across supertypes {sts}: {alleles}")

    # Eligible units: every allele in the unit clears the measurement floor.
    by_supertype = {}
    for unit, alleles in units.items():
        st = SUPERTYPE[alleles[0]]
        if st == "Unclassified":
            continue
        if min(counts[a] for a in alleles) < min_measurements:
            continue
        by_supertype.setdefault(st, []).append(unit)

    rng = np.random.default_rng(seed)
    test_units = set()
    held_out, train_only = {}, []
    for st in sorted(by_supertype):
        eligible = sorted(by_supertype[st])
        if len(eligible) < 2:
            train_only.append(st)
            continue
        k = max(1, int(round(test_size * len(eligible))))
        chosen = rng.choice(eligible, size=k, replace=False)
        test_units |= set(chosen)
        held_out[st] = len(chosen)

    is_test = df["hla_pseudoseq"].isin(test_units).to_numpy()
    test_pos = np.flatnonzero(is_test)
    train_pos = np.flatnonzero(~is_test)

    if verbose:
        n_test_alleles = df.iloc[test_pos]["allele"].nunique()
        print(f"[*] Split: {len(train_pos):,} train / {len(test_pos):,} test rows "
              f"({len(test_pos) / len(df):.1%} test)")
        print(f"    {n_test_alleles} held-out alleles across {len(held_out)} supertypes: "
              + ", ".join(f"{st}({n})" for st, n in sorted(held_out.items())))
        if train_only:
            print(f"    train-only supertypes (too few eligible alleles): "
                  f"{', '.join(train_only)}")
        smallest = df.iloc[test_pos].groupby("allele").size().min()
        print(f"    smallest held-out allele: {smallest} measurements")

    return train_pos, test_pos


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StabilityNet(nn.Module):
    """Single hidden layer, sigmoid units, sigmoid output -- the Nielsen et al. lineage."""

    def __init__(self, in_features=860, hidden_dim=60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net(x)).squeeze(-1)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(X_train, y_train, args):
    """Train for a fixed number of epochs. No validation set, no early stopping."""
    device = resolve_device(args.device)
    print(f"[*] Training on {device} for {args.epochs} epochs")

    model = StabilityNet(in_features=X_train.shape[1], hidden_dim=args.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bx)
        total /= len(ds)
        scheduler.step(total)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"    epoch {epoch:3d}/{args.epochs} | train MSE {total:.5f}")

    def predict_fn(X):
        model.eval()
        loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                            batch_size=args.batch_size, shuffle=False)
        out = []
        with torch.no_grad():
            for (bx,) in loader:
                out.append(model(bx.to(device)).cpu().numpy())
        return np.concatenate(out)

    return model, predict_fn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(alleles, thalf_hours, y_true, y_pred):
    """
    Mean per-allele Spearman is the headline. Everything else is context.
    """
    per_allele = []
    frame = pd.DataFrame({"allele": alleles, "true": y_true, "pred": y_pred})
    for allele, g in frame.groupby("allele"):
        if len(g) < 3 or g["true"].std() < 1e-6 or g["pred"].std() < 1e-6:
            continue
        per_allele.append({
            "allele": allele,
            "n": len(g),
            "scc": spearmanr(g["true"], g["pred"]).statistic,
            "pcc": pearsonr(g["true"], g["pred"])[0],
        })
    per_allele = pd.DataFrame(per_allele)

    metrics = {
        "mean_allele_scc": float(per_allele["scc"].mean()),
        "mean_allele_pcc": float(per_allele["pcc"].mean()),
        "median_allele_scc": float(per_allele["scc"].median()),
        "global_scc": float(spearmanr(y_true, y_pred).statistic),
        "n_alleles_scored": int(len(per_allele)),
    }
    return metrics, per_allele.sort_values("scc", ascending=False).reset_index(drop=True)


def peptide_only_null(train_df, test_df, y_train):
    """
    Null baseline: predict each peptide's mean training score, ignoring the allele.

    This matters because Rasmussen built one ~383-peptide panel per motif group and
    screened every allele in that group against it, so ~90-98% of test peptides were
    already seen paired with other alleles. A model that has learnt nothing about
    alleles still scores non-trivially here; the real model must beat this.
    """
    peptide_mean = pd.Series(y_train).groupby(train_df["peptide"].to_numpy()).mean()
    return test_df["peptide"].map(peptide_mean).fillna(y_train.mean()).to_numpy()


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--encoding", default="blosum50", choices=["blosum50", "sparse"])
    p.add_argument("--t0", type=float, default=1.0,
                   help="Rescaling threshold in hours")
    p.add_argument("--test_size", type=float, default=0.25,
                   help="Fraction of eligible alleles held out per supertype")
    p.add_argument("--min_measurements", type=int, default=100,
                   help="Alleles with fewer rows are never held out")
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=90,
                   help="Fixed epoch count; there is no early stopping")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.data_path):
        sys.exit(f"[!] Dataset not found: {args.data_path}\n"
                 f"    Run from the repo root, or pass --data_path.")
    df = pd.read_csv(args.data_path).reset_index(drop=True)
    print(f"[*] Loaded {len(df):,} measurements over {df['allele'].nunique()} alleles")

    missing = set(df["allele"]) - set(SUPERTYPE)
    if missing:
        sys.exit(f"[!] No supertype assignment for: {sorted(missing)}")

    train_pos, test_pos = split_by_supertype(
        df, test_size=args.test_size, min_measurements=args.min_measurements,
        seed=args.seed,
    )
    train_df, test_df = df.iloc[train_pos], df.iloc[test_pos]

    X_train = encode_sequences(train_df["peptide"].values,
                               train_df["hla_pseudoseq"].values, args.encoding)
    X_test = encode_sequences(test_df["peptide"].values,
                              test_df["hla_pseudoseq"].values, args.encoding)
    y_train = transform_target(train_df["thalf_hours"].values, args.t0)
    y_test = transform_target(test_df["thalf_hours"].values, args.t0)
    print(f"[*] Encoded with {args.encoding}: {X_train.shape[1]} features")

    model, predict_fn = train_model(X_train, y_train, args)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "baseline_model.pt"))

    y_pred = predict_fn(X_test)
    alleles = test_df["allele"].to_numpy()
    thalf = test_df["thalf_hours"].to_numpy()

    metrics, per_allele = evaluate(alleles, thalf, y_test, y_pred)
    null_pred = peptide_only_null(train_df, test_df, y_train)
    null_metrics, _ = evaluate(alleles, thalf, y_test, null_pred)
    metrics["null_peptide_only_scc"] = null_metrics["mean_allele_scc"]
    metrics["lift_over_null"] = metrics["mean_allele_scc"] - null_metrics["mean_allele_scc"]

    print("\n" + "=" * 58)
    print(f"  mean per-allele SCC   {metrics['mean_allele_scc']:.4f}   <- headline")
    print(f"  peptide-only null     {metrics['null_peptide_only_scc']:.4f}")
    print(f"  lift over null        {metrics['lift_over_null']:+.4f}")
    print("-" * 58)
    print(f"  mean per-allele PCC   {metrics['mean_allele_pcc']:.4f}")
    print(f"  median allele SCC     {metrics['median_allele_scc']:.4f}")
    print(f"  global SCC            {metrics['global_scc']:.4f}")
    print(f"  alleles scored        {metrics['n_alleles_scored']}")
    print("=" * 58)

    out = test_df.copy()
    out["true_score"], out["pred_score"] = y_test, y_pred
    out.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    per_allele.to_csv(os.path.join(args.output_dir, "allele_metrics.csv"), index=False)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[*] Wrote predictions, per-allele metrics and metrics.json to {args.output_dir}/")

    if args.plot:
        plot_results(per_allele, metrics, args.output_dir)
    print(f"[*] Done in {time.time() - start:.1f}s")


def plot_results(per_allele, metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, HUE, ACC = "#1c1c1c", "#6b6b6b", "#dcdcdc", "#2b7a8c", "#b5485d"
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(per_allele) + 1.6))
    d = per_allele.sort_values("scc")
    ax.barh(d["allele"], d["scc"], color=HUE, height=0.7)
    ax.axvline(metrics["null_peptide_only_scc"], color=ACC, lw=1.4, ls="--",
               label=f"peptide-only null  {metrics['null_peptide_only_scc']:.3f}")
    ax.axvline(metrics["mean_allele_scc"], color=INK, lw=1.4,
               label=f"mean across alleles  {metrics['mean_allele_scc']:.3f}")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.7)
    for t in leg.get_texts():
        t.set_color(MUTED)
    ax.set_xlabel("Spearman rank correlation", color=MUTED, fontsize=9)
    ax.set_title("Per-allele ranking performance on held-out alleles",
                 color=INK, fontsize=10, loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    fig.tight_layout()
    path = os.path.join(output_dir, "baseline_evaluation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[*] Wrote {path}")


if __name__ == "__main__":
    main()
