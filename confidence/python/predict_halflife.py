"""Predict peptide-HLA half-life from Boltz-2 confidence features.

Three predictors, reported side by side rather than picking a winner:

  1. ZERO-SHOT RANK -- rank peptides within an allele by a PRE-REGISTERED
     confidence feature. Uses no labels at all. Produces an ordering, not hours.

  2. CALIBRATED ZERO-SHOT -- the same single feature, mapped to hours by an
     isotonic (monotone) fit on TRAINING alleles only. The ranking is unchanged
     by construction; calibration only assigns hours to it. This uses labels,
     so it is not zero-shot -- it is the minimum-assumption way to get hours.

  3. TRAINED -- all confidence features into a supervised model, predicting
     s = 2^(-t0/t_half), inverted to hours. Two model families are reported
     (the repo's StabilityNet and a gradient-boosted tree) because neither was
     chosen in advance and reporting only the better one would be selection.

Evaluation is leave-ONE-allele-out: the held-out allele never appears in
training or in the calibration fit. Headline metric is mean per-allele Spearman,
matching the repo convention.

PRE-REGISTRATION. PREREG_FEATURES below was fixed after the 150-complex trimer
run and BEFORE any later data was folded. The reason this matters: the best
feature from the earlier 191-token run (pocketB_plddt_mean) did NOT replicate
in the trimer geometry, so feature choice on this problem has already been
shown to overfit once. Re-selecting the best feature on each new batch would
reproduce that mistake. If you add data, do not edit this list to whatever
scores highest -- that is the whole point of writing it down.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `baseline` resolves

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PREREG_FEATURES = ["pep_plddt_min", "pep_plddt_mean"]
# Sign convention: higher pLDDT is expected to mean a longer half-life, so these
# are used as-is. A feature expected to run the other way (PAE, where higher =
# less confident) would be negated here, not flipped after seeing the result.

ID_COLS = {"allele", "peptide", "thalf_hours", "record_id", "supertype",
           "hla_seq", "hla_pseudoseq", "chain_a", "chain_b2m"}
T0 = 1.0


def s_to_hours(s):
    s = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-9)
    return -T0 / np.log2(s)


def hours_to_s(h):
    h = np.asarray(h, dtype=float)
    out = np.zeros_like(h)
    m = h > 0
    out[m] = 2.0 ** (-T0 / h[m])
    return out


def per_allele(df, pred_col, truth_col="thalf_hours"):
    rs = {}
    for a, g in df.groupby("allele"):
        x, y = g[pred_col].to_numpy(), g[truth_col].to_numpy()
        rs[a] = (spearmanr(x, y).statistic
                 if np.std(x) > 0 and np.std(y) > 0 and len(g) >= 5 else np.nan)
    v = np.array([r for r in rs.values() if np.isfinite(r)])
    return (v.mean() if len(v) else np.nan,
            v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, rs)


def summarise(df, pred_col, label, hours_col=None):
    mu, se, rs = per_allele(df, pred_col)
    out = {"predictor": label, "mean_allele_scc": mu, "se": se,
           "n_alleles": int(np.isfinite(list(rs.values())).sum()),
           "global_scc": spearmanr(df[pred_col], df.thalf_hours).statistic}
    lab = (df.thalf_hours > 1.0).astype(int)
    out["auc_1h"] = (roc_auc_score(lab, df[pred_col])
                     if 0 < lab.sum() < len(lab) else np.nan)
    if hours_col:
        out["mae_hours"] = float(np.mean(np.abs(df[hours_col] - df.thalf_hours)))
        out["rmse_hours"] = float(np.sqrt(np.mean((df[hours_col] - df.thalf_hours) ** 2)))
        out["pcc_hours"] = pearsonr(df[hours_col], df.thalf_hours).statistic
    return out


def train_net(Xtr, ytr, seed, epochs, hidden_dim):
    import torch
    import torch.nn as nn
    from baseline import StabilityNet
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = StabilityNet(in_features=Xtr.shape[1], hidden_dim=hidden_dim)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = nn.MSELoss()
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)   # flat: forward() squeezes to (N,)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            lf(m(X[b]), y[b]).backward()
            opt.step()
    m.eval()

    def f(Xte):
        with torch.no_grad():
            return m(torch.tensor(Xte, dtype=torch.float32)).numpy().ravel()
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True,
                   help="one or more features.jsonl files; concatenated")
    p.add_argument("--out", default="predict")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.DataFrame([json.loads(l) for l in Path(f).read_text().splitlines()
                                  if l.strip()]) for f in args.features],
                   ignore_index=True).drop_duplicates(["allele", "peptide"])
    feats = [c for c in df.columns if c not in ID_COLS
             and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    missing = [f for f in PREREG_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"pre-registered feature(s) absent from input: {missing}")
    print(f"[in] {len(df)} complexes | {df.allele.nunique()} alleles | "
          f"{len(feats)} features | pre-registered: {PREREG_FEATURES}", flush=True)

    rows, keep = [], df[["allele", "peptide", "thalf_hours"]].copy()

    # ---- 1. zero-shot: no labels touched ----
    for f in PREREG_FEATURES:
        keep[f"zs_{f}"] = df[f].to_numpy()
        rows.append(summarise(df.assign(_p=df[f]), "_p", f"zero-shot rank [{f}]"))

    # ---- 2. calibrated zero-shot: isotonic fit on training alleles only ----
    for f in PREREG_FEATURES:
        hrs = np.full(len(df), np.nan)
        for a in df.allele.unique():
            tr, te = (df.allele != a).to_numpy(), (df.allele == a).to_numpy()
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(df.loc[tr, f].to_numpy(), hours_to_s(df.loc[tr, "thalf_hours"].to_numpy()))
            hrs[te] = s_to_hours(iso.predict(df.loc[te, f].to_numpy()))
        keep[f"cal_hours_{f}"] = hrs
        rows.append(summarise(df.assign(_p=hrs, _h=hrs), "_p",
                              f"calibrated zero-shot [{f}]", hours_col="_h"))

    # ---- 3. trained: all features ----
    Xraw = df[feats].to_numpy(dtype=np.float32)
    y_s = hours_to_s(df.thalf_hours.to_numpy()).astype(np.float32)

    for name in ("StabilityNet", "GBM"):
        acc = np.zeros((args.seeds, len(df)))
        for si, seed in enumerate(range(42, 42 + args.seeds)):
            pred = np.full(len(df), np.nan)
            for a in df.allele.unique():
                tr, te = (df.allele != a).to_numpy(), (df.allele == a).to_numpy()
                sc = StandardScaler().fit(Xraw[tr])
                Xtr, Xte = sc.transform(Xraw[tr]), sc.transform(Xraw[te])
                if name == "StabilityNet":
                    f = train_net(Xtr.astype(np.float32), y_s[tr], seed,
                                  args.epochs, args.hidden_dim)
                    pred[te] = f(Xte.astype(np.float32))
                else:
                    g = GradientBoostingRegressor(random_state=seed, n_estimators=300,
                                                  max_depth=2, learning_rate=0.05)
                    g.fit(Xtr, y_s[tr])
                    pred[te] = g.predict(Xte)
            acc[si] = pred
        mean_s = acc.mean(0)
        keep[f"pred_s_{name}"] = mean_s
        keep[f"pred_hours_{name}"] = s_to_hours(mean_s)
        rows.append(summarise(df.assign(_p=mean_s, _h=s_to_hours(mean_s)), "_p",
                              f"trained [{name}, {len(feats)} feats, {args.seeds} seeds]",
                              hours_col="_h"))

    res = pd.DataFrame(rows)
    res.to_csv(out / "predictor_comparison.csv", index=False)
    keep.to_csv(out / "predictions.csv", index=False)

    show = ["predictor", "mean_allele_scc", "se", "n_alleles", "global_scc",
            "auc_1h", "mae_hours", "pcc_hours"]
    print("\n=== leave-one-allele-out ===")
    print(res.reindex(columns=show).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nreference bars (canonical protocol, full data):")
    print("  peptide-only null        0.3045")
    print("  BLOSUM baseline          0.4626")


if __name__ == "__main__":
    main()
