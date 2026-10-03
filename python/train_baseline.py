#!/usr/bin/env python3
"""
NetMHCstabpan Baseline Model for Peptide-HLA Complex Stability Prediction.

Reproduces the supervised neural network baseline described in:
Rasmussen et al. (2016) "Pan-Specific Prediction of Peptide–MHC Class I Complex
Stability, a Correlate of T Cell Immunogenicity", The Journal of Immunology.

Dataset features:
  - 9-mer peptide (9 positions)
  - HLA pseudosequence (34 positions)
  Total sequence length: 43 residues -> 43 x 20 = 860 input dimensions.
  Target: thalf_hours (half-life in hours).
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Amino Acid Alphabet & BLOSUM Matrices
# ---------------------------------------------------------------------------
# Canonical 20 amino acids
AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Standard BLOSUM50 matrix (values divided by 5.0 as in Rasmussen et al. 2016)
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

# Standard BLOSUM62 matrix (values divided by 5.0)
BLOSUM62_RAW = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]
BLOSUM62_MATRIX = np.array(BLOSUM62_RAW, dtype=np.float32) / 5.0


def encode_sequences(peptides, pseudoseqs, encoding="blosum50"):
    """
    Encode peptide (9 aa) and HLA pseudosequence (34 aa) into flat feature vectors.
    Total output shape: (N, 43 * 20) = (N, 860).
    """
    n_samples = len(peptides)
    combined = [p + h for p, h in zip(peptides, pseudoseqs)]
    
    # Map sequence characters to integer indices (0..19)
    indices = np.zeros((n_samples, 43), dtype=np.int32)
    for i, seq in enumerate(combined):
        for j, aa in enumerate(seq):
            indices[i, j] = AA_TO_IDX.get(aa, 0)

    if encoding == "blosum50":
        features = BLOSUM50_MATRIX[indices].reshape(n_samples, -1)
    elif encoding == "blosum62":
        features = BLOSUM62_MATRIX[indices].reshape(n_samples, -1)
    elif encoding == "onehot":
        features = np.eye(20, dtype=np.float32)[indices].reshape(n_samples, -1)
    elif encoding == "sparse":
        # NetMHCpan sparse: 0.9 for matching aa, 0.05 for others
        oh = np.eye(20, dtype=np.float32)[indices].reshape(n_samples, -1)
        features = oh * 0.85 + 0.05
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Target Transformations
# ---------------------------------------------------------------------------
def transform_target(thalf_hours, mode="netmhc", t0=1.0):
    """
    Transform raw half-life (th) to model target space.
      'netmhc': s = 2^(-t0 / th)  [bounded in 0..1, t0=1.0h is paper default]
      'log1p':  y = log10(1 + th)
      'raw':    y = th
    """
    th = np.array(thalf_hours, dtype=np.float32)
    if mode == "netmhc":
        s = np.zeros_like(th)
        mask = th > 0
        s[mask] = 2.0 ** (-t0 / th[mask])
        return s
    elif mode == "log1p":
        return np.log10(1.0 + np.maximum(th, 0.0))
    elif mode == "raw":
        return th
    else:
        raise ValueError(f"Unknown target transformation: {mode}")


def inverse_transform_target(pred, mode="netmhc", t0=1.0, max_thalf=256.7):
    """
    Invert model prediction back to half-life in hours.
    """
    pred = np.array(pred, dtype=np.float32)
    if mode == "netmhc":
        # s = 2^(-t0 / th)  =>  th = -t0 / log2(s)
        th = np.zeros_like(pred)
        # Numerical protection: clamp between 1e-6 and 1 - 1e-6
        s_clamped = np.clip(pred, 1e-6, 1.0 - 1e-6)
        log2_s = np.log2(s_clamped)
        mask = log2_s < -1e-5
        th[mask] = -t0 / log2_s[mask]
        th[pred >= 0.999] = max_thalf
        th[pred <= 1e-4] = 0.0
        return np.clip(th, 0.0, max_thalf)
    elif mode == "log1p":
        return np.maximum(10.0 ** pred - 1.0, 0.0)
    elif mode == "raw":
        return np.maximum(pred, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
def calculate_metrics(y_true_thalf, y_pred_thalf, y_true_score, y_pred_score, alleles):
    """
    Calculate global & per-allele Pearson, Spearman, AUC, RMSE, MAE.
    """
    # Global correlations on transformed score and on raw half-life
    pcc_score, _ = pearsonr(y_true_score, y_pred_score)
    scc_score, _ = spearmanr(y_true_score, y_pred_score)
    pcc_thalf, _ = pearsonr(y_true_thalf, y_pred_thalf)
    scc_thalf, _ = spearmanr(y_true_thalf, y_pred_thalf)

    # Classification AUC at thresholds 1h and 2h
    binary_1h = (y_true_thalf >= 1.0).astype(int)
    auc_1h = roc_auc_score(binary_1h, y_pred_score) if len(np.unique(binary_1h)) > 1 else np.nan

    binary_2h = (y_true_thalf >= 2.0).astype(int)
    auc_2h = roc_auc_score(binary_2h, y_pred_score) if len(np.unique(binary_2h)) > 1 else np.nan

    # Regression error on half-life in hours
    rmse = np.sqrt(mean_squared_error(y_true_thalf, y_pred_thalf))
    mae = mean_absolute_error(y_true_thalf, y_pred_thalf)

    # Per-allele metrics
    allele_df = pd.DataFrame({
        "allele": alleles,
        "true_thalf": y_true_thalf,
        "pred_thalf": y_pred_thalf,
        "true_score": y_true_score,
        "pred_score": y_pred_score
    })

    allele_pccs = []
    allele_sccs = []
    allele_records = []

    for allele, grp in allele_df.groupby("allele"):
        n_grp = len(grp)
        # Need at least 3 samples and non-zero variance to compute Pearson/Spearman
        if n_grp >= 3 and grp["true_score"].std() > 1e-6 and grp["pred_score"].std() > 1e-6:
            pcc_a, _ = pearsonr(grp["true_score"], grp["pred_score"])
            scc_a, _ = spearmanr(grp["true_score"], grp["pred_score"])
            if not np.isnan(pcc_a):
                allele_pccs.append(pcc_a)
            if not np.isnan(scc_a):
                allele_sccs.append(scc_a)
            allele_records.append({
                "allele": allele,
                "n_samples": n_grp,
                "pcc": float(pcc_a),
                "scc": float(scc_a)
            })

    mean_allele_pcc = float(np.mean(allele_pccs)) if allele_pccs else np.nan
    mean_allele_scc = float(np.mean(allele_sccs)) if allele_sccs else np.nan

    return {
        "global_pcc_score": float(pcc_score),
        "global_scc_score": float(scc_score),
        "global_pcc_thalf": float(pcc_thalf),
        "global_scc_thalf": float(scc_thalf),
        "mean_allele_pcc": mean_allele_pcc,
        "mean_allele_scc": mean_allele_scc,
        "auc_1h": float(auc_1h),
        "auc_2h": float(auc_2h),
        "rmse_hours": float(rmse),
        "mae_hours": float(mae),
        "n_evaluated_alleles": len(allele_records),
    }, pd.DataFrame(allele_records)


# ---------------------------------------------------------------------------
# PyTorch Supervised Model
# ---------------------------------------------------------------------------
def train_pytorch_model(X_train, y_train, X_val, y_val, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Training PyTorch baseline on device: {device}")

    # Build model architecture matching NetMHCpan / NetMHCstabpan
    class StabilityNet(nn.Module):
        def __init__(self, in_features=860, hidden_dim=60, target_mode="netmhc"):
            super().__init__()
            self.target_mode = target_mode
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, 1),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.net(x)
            if self.target_mode == "netmhc":
                # Score is strictly in [0, 1]
                return self.sigmoid(out).squeeze(-1)
            else:
                return out.squeeze(-1)

    model = StabilityNet(
        in_features=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        target_mode=args.target_transform
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"    Epoch {epoch:02d}/{args.epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Predict function
    def predict_fn(X):
        model.eval()
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(device)
                p = model(bx).cpu().numpy()
                preds.append(p)
        return np.concatenate(preds)

    return model, predict_fn


# ---------------------------------------------------------------------------
# Splitting Strategies
# ---------------------------------------------------------------------------
def get_train_test_split(df, strategy="random", test_size=0.2, seed=42):
    """
    Implements splitting strategies:
      - 'random': Standard random stratified/shuffled split
      - 'allele': Leave-HLA-alleles-out split (tests generalization to unseen HLA alleles)
      - 'cluster': Peptide sequence dissimilarity split
    """
    np.random.seed(seed)
    if strategy == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=seed, shuffle=True
        )
    elif strategy == "allele":
        unique_alleles = df["allele"].unique()
        n_test_alleles = max(1, int(len(unique_alleles) * test_size))
        test_alleles = np.random.choice(unique_alleles, size=n_test_alleles, replace=False)
        test_mask = df["allele"].isin(test_alleles)
        train_idx = df[~test_mask].index.values
        test_idx = df[test_mask].index.values
        print(f"[*] Leave-allele-out split: {len(test_alleles)} held-out alleles in test set.")
    elif strategy == "cluster":
        # Group by unique 9-mer peptides to avoid train/test peptide overlap
        unique_peptides = df["peptide"].unique()
        np.random.shuffle(unique_peptides)
        n_test_pep = int(len(unique_peptides) * test_size)
        test_peps = set(unique_peptides[:n_test_pep])
        test_mask = df["peptide"].isin(test_peps)
        train_idx = df[~test_mask].index.values
        test_idx = df[test_mask].index.values
        print(f"[*] Peptide-grouped split: {len(test_peps)} unique peptides held out.")
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NetMHCstabpan Baseline Reproduction Script")
    parser.add_argument("--data_path", type=str, default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv",
                        help="Path to CSV dataset")
    parser.add_argument("--encoding", type=str, default="blosum50", choices=["blosum50", "blosum62", "onehot", "sparse"],
                        help="Sequence encoding method (default: blosum50)")
    parser.add_argument("--target_transform", type=str, default="netmhc", choices=["netmhc", "log1p", "raw"],
                        help="Target transformation mode (default: netmhc, s = 2^(-t0/th))")
    parser.add_argument("--t0", type=float, default=1.0, help="Threshold t0 in hours for NetMHC transformation (default: 1.0)")
    parser.add_argument("--split_strategy", type=str, default="random", choices=["random", "allele", "cluster"],
                        help="Data split strategy: random, allele (leave-allele-out), or cluster (peptide-grouped)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for test split (default: 0.2)")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of train data for validation (default: 0.1)")
    parser.add_argument("--hidden_dim", type=int, default=60, help="Hidden units in ANN (default: 60)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--device", type=str, default="auto", help="Device for PyTorch: auto, mps, cuda, or cpu")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save predictions and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate evaluation plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("      NetMHCstabpan Supervised Neural Network Baseline")
    print("=" * 70)
    print(f"Data file:         {args.data_path}")
    print(f"Sequence encoding: {args.encoding}")
    print(f"Target transform:  {args.target_transform} (t0 = {args.t0}h)")
    print(f"Split strategy:    {args.split_strategy} (test_size = {args.test_size})")
    print(f"Random seed:       {args.seed}")
    print("=" * 70)

    # 1. Load data
    t_start = time.time()
    if not os.path.isfile(args.data_path):
        print(f"[!] Error: File '{args.data_path}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    print(f"[*] Loaded dataset with {len(df)} rows across {df['allele'].nunique()} HLA alleles.")

    # 2. Split data
    train_idx, test_idx = get_train_test_split(df, strategy=args.split_strategy, test_size=args.test_size, seed=args.seed)
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Sub-split train into train & validation
    subtrain_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=args.val_size, random_state=args.seed, shuffle=True
    )
    subtrain_df = train_df.iloc[subtrain_idx].copy().reset_index(drop=True)
    val_df = train_df.iloc[val_idx].copy().reset_index(drop=True)

    print(f"[*] Dataset split: Train={len(subtrain_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 3. Feature Encoding
    print(f"[*] Encoding sequences using '{args.encoding}'...")
    X_train = encode_sequences(subtrain_df["peptide"].values, subtrain_df["hla_pseudoseq"].values, encoding=args.encoding)
    X_val = encode_sequences(val_df["peptide"].values, val_df["hla_pseudoseq"].values, encoding=args.encoding)
    X_test = encode_sequences(test_df["peptide"].values, test_df["hla_pseudoseq"].values, encoding=args.encoding)
    print(f"[*] Feature dimension: {X_train.shape[1]} (43 residues x 20 features)")

    # 4. Target Transformation
    y_train = transform_target(subtrain_df["thalf_hours"].values, mode=args.target_transform, t0=args.t0)
    y_val = transform_target(val_df["thalf_hours"].values, mode=args.target_transform, t0=args.t0)
    y_test_score = transform_target(test_df["thalf_hours"].values, mode=args.target_transform, t0=args.t0)
    y_test_thalf = test_df["thalf_hours"].values

    # 5. Train Model
    import torch

    model, predict_fn = train_pytorch_model(X_train, y_train, X_val, y_val, args)
    model_save_path = os.path.join(args.output_dir, "baseline_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[*] Saved model checkpoint to: {model_save_path}")

    # 6. Predict on Test Set
    print("[*] Generating test set predictions...")
    y_pred_score = predict_fn(X_test)
    y_pred_thalf = inverse_transform_target(y_pred_score, mode=args.target_transform, t0=args.t0)

    # 7. Evaluate Metrics
    metrics, allele_metrics = calculate_metrics(
        y_true_thalf=y_test_thalf,
        y_pred_thalf=y_pred_thalf,
        y_true_score=y_test_score,
        y_pred_score=y_pred_score,
        alleles=test_df["allele"].values
    )

    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS (TEST SET)")
    print("=" * 70)
    print(f"  Mean Per-Allele Pearson Correlation (PCC):  {metrics['mean_allele_pcc']:.4f}")
    print(f"  Mean Per-Allele Spearman Correlation (SCC): {metrics['mean_allele_scc']:.4f}")
    print(f"  Global Pearson Correlation (PCC):           {metrics['global_pcc_score']:.4f}")
    print(f"  Global Spearman Correlation (SCC):          {metrics['global_scc_score']:.4f}")
    print(f"  Classification AUC (t1/2 >= 1h):            {metrics['auc_1h']:.4f}")
    print(f"  Classification AUC (t1/2 >= 2h):            {metrics['auc_2h']:.4f}")
    print(f"  Root Mean Squared Error (RMSE in hours):    {metrics['rmse_hours']:.2f}")
    print(f"  Mean Absolute Error (MAE in hours):         {metrics['mae_hours']:.2f}")
    print(f"  Alleles Evaluated:                          {metrics['n_evaluated_alleles']}")
    print("=" * 70)

    # 8. Save Outputs
    # Save predictions dataframe
    test_df["true_thalf_hours"] = y_test_thalf
    test_df["pred_thalf_hours"] = y_pred_thalf
    test_df["true_score"] = y_test_score
    test_df["pred_score"] = y_pred_score
    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    test_df.to_csv(preds_path, index=False)
    print(f"[*] Saved test predictions to: {preds_path}")

    # Save per-allele metrics
    allele_metrics_path = os.path.join(args.output_dir, "allele_metrics.csv")
    allele_metrics.to_csv(allele_metrics_path, index=False)
    print(f"[*] Saved per-allele metrics to: {allele_metrics_path}")

    # Save summary metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[*] Saved metrics JSON to: {metrics_path}")

    # 9. Plotting (if matplotlib available and enabled)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

            # Subplot 1: Score correlation scatter plot
            ax = axes[0]
            scatter = ax.scatter(y_test_score, y_pred_score, alpha=0.35, s=12, c="teal", edgecolors="none")
            ax.plot([0, 1], [0, 1], color="crimson", linestyle="--", linewidth=1.5, label="Identity")
            ax.set_title(f"Stability Score Correlation\nGlobal PCC: {metrics['global_pcc_score']:.3f} | Mean Allele PCC: {metrics['mean_allele_pcc']:.3f}")
            ax.set_xlabel("True Transformed Score (s)")
            ax.set_ylabel("Predicted Transformed Score (s)")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left")

            # Subplot 2: Per-allele PCC distribution
            ax = axes[1]
            if len(allele_metrics) > 0:
                ax.hist(allele_metrics["pcc"], bins=20, color="steelblue", edgecolor="black", alpha=0.8)
                ax.axvline(metrics["mean_allele_pcc"], color="crimson", linestyle="--", linewidth=2,
                           label=f"Mean: {metrics['mean_allele_pcc']:.3f}")
                ax.set_title("Distribution of Per-Allele Pearson Correlations")
                ax.set_xlabel("Pearson Correlation (r)")
                ax.set_ylabel("Number of Alleles")
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.legend(loc="upper left")

            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, "baseline_evaluation.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"[*] Saved evaluation plots to: {plot_path}")
        except Exception as e:
            print(f"[!] Warning: Could not generate plots: {e}")

    print(f"[*] Total execution time: {time.time() - t_start:.2f}s")
    print("[*] Baseline execution completed successfully.")


if __name__ == "__main__":
    main()
