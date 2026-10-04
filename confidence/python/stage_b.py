"""Stage B: structural features -> predicted peptide-HLA half-life in hours.

Takes the label-free per-complex feature cache written by fold_complexes.py /
fold_trimer.py and runs the supervised half of the pipeline end to end:

    features + labels -> split -> model -> predicted s -> half-life in hours -> metrics

Model, loss, target transform and metrics are IMPORTED UNCHANGED from the repo's
baseline.py, per its stated project-wide convention. Only the feature function
differs between arms, which is the same discipline run_ablation.py used.

Arms:
    blosum          860 dims   the baseline, reproduced here for reference
    struct           ~34 dims  structural confidence scalars alone
    blosum+struct   ~894 dims  the augmentation arm -- the open question, and a
                               2.9% width increase rather than the 89% that the
                               failed 1,628-dim Boltz-embedding arm imposed

SPLIT CAVEAT, stated loudly because it governs how these numbers may be read:
baseline.py's split_by_supertype requires >=100 measurements per evaluated
allele. A 300-complex stage-1 cache has ~30 rows per allele, so the canonical
split cannot be formed. This script therefore uses leave-ONE-allele-out CV over
whatever alleles are present. The held-out allele is never in training, so the
generalisation question is the same one -- but the training set is ~250 rows
against the canonical protocol's 21,943, so absolute values here are NOT
comparable to the repo's 0.4626 and must not be quoted as if they were. What IS
readable is the DIFFERENCE between arms on identical folds.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from baseline import (BLOSUM50_MATRIX, AA_TO_IDX, transform_target,  # noqa: E402
                      StabilityNet, resolve_device)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ID_COLS = {"allele", "peptide", "thalf_hours", "record_id", "supertype",
           "hla_seq", "hla_pseudoseq", "chain_a", "chain_b2m"}
T0 = 1.0


def blosum_block(seqs: pd.Series) -> np.ndarray:
    idx = np.zeros((len(seqs), len(seqs.iloc[0])), dtype=np.int32)
    for i, s in enumerate(seqs):
        for j, aa in enumerate(s):
            idx[i, j] = AA_TO_IDX.get(aa, 0)
    return BLOSUM50_MATRIX[idx].reshape(len(seqs), -1).astype(np.float32)


def train_once(Xtr, ytr, hidden_dim, epochs, seed, device):
    """Baseline recipe: MSE on s, Adam lr 1e-3, wd 1e-5, batch 128."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StabilityNet(in_features=Xtr.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xtr, device=device)
    # StabilityNet.forward squeezes to (N,), so the target must be flat too --
    # a (N,1) target broadcasts into an (N,N) loss and silently trains garbage.
    yt = torch.tensor(ytr, device=device)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model, X, device):
    with torch.no_grad():
        return model(torch.tensor(X, device=device)).cpu().numpy().ravel()


def s_to_hours(s: np.ndarray) -> np.ndarray:
    """Invert s = 2^(-t0/th)  ->  th = -t0 / log2(s). s<=0 -> 0, s>=1 -> inf-capped."""
    s = np.clip(s, 1e-6, 1 - 1e-9)
    return -T0 / np.log2(s)


def metrics(df: pd.DataFrame) -> dict:
    """Mirror baseline.py's evaluate(): per-allele and global, score and hours."""
    pa_p, pa_s = [], []
    for _, g in df.groupby("allele"):
        if g.y_true.std() == 0 or g.y_pred.std() == 0 or len(g) < 5:
            continue
        pa_p.append(pearsonr(g.y_true, g.y_pred).statistic)
        pa_s.append(spearmanr(g.y_true, g.y_pred).statistic)
    out = {
        "mean_allele_pcc": float(np.mean(pa_p)) if pa_p else np.nan,
        "mean_allele_scc": float(np.mean(pa_s)) if pa_s else np.nan,
        "sd_allele_scc": float(np.std(pa_s, ddof=1)) if len(pa_s) > 1 else np.nan,
        "n_alleles_scored": len(pa_s),
        "global_pcc_score": float(pearsonr(df.y_true, df.y_pred).statistic),
        "global_scc_score": float(spearmanr(df.y_true, df.y_pred).statistic),
        "global_scc_thalf": float(spearmanr(df.thalf_hours, df.thalf_pred).statistic),
        "rmse_hours": float(np.sqrt(np.mean((df.thalf_hours - df.thalf_pred) ** 2))),
        "mae_hours": float(np.mean(np.abs(df.thalf_hours - df.thalf_pred))),
    }
    for thr in (1.0, 2.0):
        lab = (df.thalf_hours > thr).astype(int)
        out[f"auc_{int(thr)}h"] = (float(roc_auc_score(lab, df.y_pred))
                                  if 0 < lab.sum() < len(lab) else np.nan)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--dataset", required=True, help="CSV with allele,peptide,hla_pseudoseq")
    p.add_argument("--out", default="stageb")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--hidden_dim", type=int, default=60)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cpu")

    feat = pd.DataFrame([json.loads(l) for l in Path(args.features).read_text().splitlines() if l.strip()])
    data = pd.read_csv(args.dataset)[["allele", "peptide", "hla_pseudoseq"]].drop_duplicates()
    df = feat.merge(data, on=["allele", "peptide"], how="inner", validate="one_to_one")
    assert len(df) == len(feat), f"merge lost rows: {len(feat)} -> {len(df)}"

    scols = [c for c in df.columns if c not in ID_COLS and df[c].nunique() > 1]
    df["y_true"] = transform_target(df.thalf_hours.values, T0)

    Xb = np.hstack([blosum_block(df.peptide), blosum_block(df.hla_pseudoseq)])
    Xs_raw = df[scols].to_numpy(dtype=np.float32)
    print(f"[in] {len(df)} complexes | {df.allele.nunique()} alleles | "
          f"blosum {Xb.shape[1]}d | struct {Xs_raw.shape[1]}d", flush=True)

    arms = {"blosum": ("b", None), "struct": ("s", None), "blosum+struct": ("bs", None)}
    rows, preds = [], []

    for arm in arms:
        for seed in range(42, 42 + args.seeds):
            fold = []
            for a in df.allele.unique():
                tr = (df.allele != a).to_numpy()
                te = ~tr
                # z-score the structural block on TRAIN ONLY; BLOSUM stays raw
                mu, sd = Xs_raw[tr].mean(0), Xs_raw[tr].std(0)
                sd[sd == 0] = 1.0
                Xs = (Xs_raw - mu) / sd
                X = {"blosum": Xb, "struct": Xs,
                     "blosum+struct": np.hstack([Xb, Xs])}[arm].astype(np.float32)
                m = train_once(X[tr], df.y_true.values[tr].astype(np.float32),
                               args.hidden_dim, args.epochs, seed, device)
                yp = predict(m, X[te], device)
                fold.append(pd.DataFrame({
                    "arm": arm, "seed": seed, "allele": a,
                    "peptide": df.peptide.values[te],
                    "thalf_hours": df.thalf_hours.values[te],
                    "y_true": df.y_true.values[te], "y_pred": yp,
                    "thalf_pred": s_to_hours(yp)}))
            fd = pd.concat(fold, ignore_index=True)
            mt = metrics(fd)
            mt.update(arm=arm, seed=seed, dims=X.shape[1])
            rows.append(mt)
            preds.append(fd)
            print(f"[{arm:>14} s{seed}] mean_allele_scc {mt['mean_allele_scc']:+.4f} "
                  f"| global_scc {mt['global_scc_score']:+.4f} "
                  f"| auc1h {mt['auc_1h']:.3f} | mae_h {mt['mae_hours']:.2f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_per_seed.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    num = res.select_dtypes("number").columns.drop(["seed"])
    summ = res.groupby("arm")[list(num)].agg(["mean", "std"])
    summ.to_csv(out / "metrics_summary.csv")

    print("\n=== mean over seeds (leave-one-allele-out) ===")
    show = ["mean_allele_scc", "mean_allele_pcc", "global_scc_score",
            "auc_1h", "auc_2h", "rmse_hours", "mae_hours", "dims"]
    print(res.groupby("arm")[show].mean().round(4).to_string())

    base = res[res.arm == "blosum"].set_index("seed").mean_allele_scc
    for arm in ("struct", "blosum+struct"):
        d = (res[res.arm == arm].set_index("seed").mean_allele_scc - base)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        print(f"[paired] {arm:>14} - blosum: {d.mean():+.4f}"
              + (f"  (t={d.mean()/se:+.2f})" if se and se > 0 else ""))


if __name__ == "__main__":
    main()
