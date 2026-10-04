"""Stage-1 analysis: does Boltz-2 confidence carry peptide-specific stability signal?

Two questions, in priority order:

  1. VARIANCE. Does each feature vary ACROSS peptides WITHIN an allele? If Boltz is
     uniformly confident about every peptide in a given groove, the feature is a
     per-allele constant, per-allele Spearman is ~0 by construction, and no model
     can recover anything. Reported as within/between SD ratio.

  2. ZERO-SHOT RANKING. Per-allele Spearman of each raw feature against thalf_hours,
     averaged over alleles. No training, no calibration -- rank correlation is
     invariant to monotone transforms, which is what makes the claim zero-shot.
     Spearman also handles the ties at thalf == 0 that would drag Pearson around.

The bar is the peptide-only null (mean per-allele SCC 0.3045 under the canonical
protocol), NOT the supervised baseline's 0.4626. A zero-shot structural score that
clears the null is extracting real peptide-HLA interaction signal.

Note: the trained arm is NOT runnable on stage-1 data -- each allele has only 30
rows, below evaluate()'s >=100-measurement floor, so split_by_supertype cannot form
a comparable split. That waits for stage 2.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ID_COLS = {"allele", "peptide", "thalf_hours", "record_id", "supertype"}


def per_allele_spearman(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean and SD of within-allele Spearman, over alleles with usable variance."""
    rs = []
    for _, g in df.groupby("allele"):
        x, y = g[col].to_numpy(), g["thalf_hours"].to_numpy()
        if np.std(x) == 0 or np.std(y) == 0 or len(g) < 5:
            continue
        r = spearmanr(x, y).statistic
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return np.nan, np.nan, 0
    return float(np.mean(rs)), float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0, len(rs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="stage1")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features) if args.features.endswith(".parquet") \
        else pd.DataFrame([json.loads(l) for l in Path(args.features).read_text().splitlines() if l.strip()])

    feats = [c for c in df.columns if c not in ID_COLS
             and pd.api.types.is_numeric_dtype(df[c])]
    print(f"[in] {len(df)} complexes, {df.allele.nunique()} alleles, {len(feats)} features")

    rows = []
    for c in feats:
        s = df[c]
        if s.isna().all() or s.nunique() <= 1:
            rows.append({"feature": c, "status": "constant/empty"})
            continue
        # variance decomposition
        within = df.groupby("allele")[c].std().mean()
        between = df.groupby("allele")[c].mean().std()
        ratio = within / between if between and np.isfinite(between) and between > 0 else np.inf
        r, rsd, n = per_allele_spearman(df, c)
        rows.append({
            "feature": c, "status": "ok",
            "mean": float(s.mean()), "sd": float(s.std()),
            "within_allele_sd": float(within) if np.isfinite(within) else np.nan,
            "between_allele_sd": float(between) if np.isfinite(between) else np.nan,
            "within_over_between": float(ratio) if np.isfinite(ratio) else np.nan,
            "per_allele_scc": r, "per_allele_scc_sd": rsd, "n_alleles": n,
            "abs_scc": abs(r) if np.isfinite(r) else np.nan,
        })

    res = pd.DataFrame(rows).sort_values("abs_scc", ascending=False, na_position="last")
    res.to_csv(out / "feature_report.csv", index=False)

    ok = res[res.status == "ok"]
    print("\n=== top features by |mean per-allele SCC| ===")
    cols = ["feature", "per_allele_scc", "per_allele_scc_sd", "n_alleles",
            "within_allele_sd", "between_allele_sd", "within_over_between"]
    print(ok[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== variance check (within-allele variation must be non-trivial) ===")
    flat = ok[ok.within_over_between < 0.1]
    print(f"features that are effectively per-allele constants "
          f"(within/between < 0.1): {len(flat)}/{len(ok)}")
    if len(flat):
        print("  ", ", ".join(flat.feature.head(12)))

    best = ok.iloc[0] if len(ok) else None
    if best is not None:
        print(f"\n[verdict] best single feature: {best.feature}  "
              f"per-allele SCC {best.per_allele_scc:+.4f} +/- {best.per_allele_scc_sd:.4f} "
              f"over {int(best.n_alleles)} alleles")
        print(f"[verdict] peptide-only null to beat (canonical protocol): 0.3045")
        print(f"[verdict] supervised baseline (canonical protocol):       0.4626")

    # per-allele detail for the single best feature
    if best is not None:
        det = []
        for a, g in df.groupby("allele"):
            x, y = g[best.feature].to_numpy(), g["thalf_hours"].to_numpy()
            r = spearmanr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
            det.append({"allele": a, "n": len(g), "scc": r,
                        "feat_sd": float(np.std(x)),
                        "thalf_median": float(np.median(y)),
                        "frac_zero": float((y == 0).mean())})
        pd.DataFrame(det).to_csv(out / "per_allele_detail.csv", index=False)
        print("\n=== per-allele detail, best feature ===")
        print(pd.DataFrame(det).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
