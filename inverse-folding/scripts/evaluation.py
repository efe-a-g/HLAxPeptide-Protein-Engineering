"""Tie-aware within-allele evaluation; explicit coverage and unavailable cells."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

EPSILON = .05
NOISE_CEILING = 'Replicate noise ceiling: unavailable (no repeated allele–peptide pairs)'
KEYS = ['representation', 'split', 'loss']

def metric_values(frame):
    valid = np.isfinite(frame.prediction)
    g = frame.loc[valid]
    n = len(g)
    if n < 3 or g.thalf_hours.nunique() < 2:
        rho, pear = np.nan, np.nan
    elif g.prediction.nunique() < 2:
        # Spearman of a constant is mathematically undefined. Operational zero
        # records absence of ranking, as requested, and is documented in report.
        rho, pear = 0., 0.
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rho = float(spearmanr(g.thalf_hours, g.prediction).statistic)
            pear = float(pearsonr(np.log(g.thalf_hours + EPSILON), g.prediction).statistic)
    binary = g.thalf_hours > 2
    auc = float(roc_auc_score(binary, g.prediction)) if binary.nunique() == 2 else np.nan
    return {'spearman': rho, 'pearson_log': pear, 'auc_gt_2h': auc}, n

def evaluate_predictions(predictions, outdir):
    """Input columns: row_id, allele, thalf_hours, prediction, representation,
    split, loss, fold. Predictions are on the log scale or an uncalibrated score.
    Writes long-form results.csv and per_allele_metrics.csv. Returns both frames.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records, per = [], []
    for key, allg in predictions.groupby(KEYS, dropna=False):
        meta = dict(zip(KEYS, key))
        assert not allg.row_id.duplicated().any(), f'Duplicate evaluation predictions: {meta}'
        for subset in ['all', 'nonzero']:
            g = allg if subset == 'all' else allg[allg.thalf_hours > 0]
            allele_rows = []
            for allele, ag in g.groupby('allele'):
                vals, n = metric_values(ag)
                for metric, val in vals.items():
                    row = {**meta, 'subset': subset, 'allele': allele, 'metric': metric,
                           'value': val, 'n': n, 'n_total': len(ag), 'constant_prediction': ag.prediction.dropna().nunique() == 1}
                    per.append(row)
                    allele_rows.append(row)
            tab = pd.DataFrame(allele_rows)
            pooled, n = metric_values(g)
            for metric, value in pooled.items():
                records.append({**meta, 'subset': subset, 'aggregation': 'pooled', 'metric': metric,
                                'value': value, 'n': n, 'n_total': len(g), 'n_alleles': g.allele.nunique(),
                                'status': 'ok' if np.isfinite(value) else 'unavailable'})
                for agg in ['macro', 'size_weighted']:
                    a = tab[(tab.metric == metric) & tab.value.notna()] if len(tab) else tab
                    val = (np.average(a.value, weights=a.n if agg == 'size_weighted' else None) if len(a) else np.nan)
                    records.append({**meta, 'subset': subset, 'aggregation': agg, 'metric': metric,
                                    'value': val, 'n': int(a.n.sum()) if len(a) else 0, 'n_total': len(g),
                                    'n_alleles': len(a), 'status': 'ok' if np.isfinite(val) else 'unavailable'})
    results, alleles = pd.DataFrame(records), pd.DataFrame(per)
    results.to_csv(outdir / 'results.csv', index=False)
    alleles.to_csv(outdir / 'per_allele_metrics.csv', index=False)
    return results, alleles

def paired_allele_bootstrap(per_allele, baseline, variant, subset='nonzero', split='leave_one_allele_out', loss='mse', seed=20261003, draws=5000):
    g = per_allele[(per_allele.subset == subset) & (per_allele.split == split) &
                   (per_allele.metric == 'spearman') & (per_allele.loss == loss)]
    p = g.pivot(index='allele', columns='representation', values='value')
    p = p[[baseline, variant]].dropna()
    delta = (p[variant] - p[baseline]).to_numpy()
    if len(delta) == 0:
        return {}
    rng = np.random.default_rng(seed)
    boot = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {'baseline': baseline, 'variant': variant, 'subset': subset, 'n_alleles': len(delta),
            'delta_macro_spearman': float(delta.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'unit': 'allele', 'draws': draws, 'seed': seed,
            'caveat': 'Descriptive allele bootstrap; related alleles and overlapping training folds are not independent; excludes training-seed uncertainty.'}
