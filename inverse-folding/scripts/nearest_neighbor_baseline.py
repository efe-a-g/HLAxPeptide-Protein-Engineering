"""Cheap Hamming nearest-neighbour stability transfer under existing LOAO.

Distances and tie rules are fixed without target-driven tuning. No model fitting.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from splits import load_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/followup/pseudoseq_if/nearest_neighbor'
METHODS = {
    'raw_1nn': 'Hamming sum, copy one nearest row',
    'raw_tie_mean': 'Hamming sum, average equally nearest rows',
    'balanced_1nn': 'Length-normalized Hamming, copy one nearest row',
    'balanced_tie_mean': 'Length-normalized Hamming, average equally nearest rows',
}


def encode(sequences):
    return np.array([list(s.encode('ascii')) for s in sequences], dtype=np.uint8)


def hamming_matrix(sequences):
    # Compact integer cache, rather than a row-by-row distance matrix.
    n, length = sequences.shape
    dtype = np.uint8 if length <= 255 else np.uint16
    distances = np.zeros((n, n), dtype=dtype)
    for position in range(length):
        distances += sequences[:, position, None] != sequences[None, :, position]
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hla', choices=['full', 'pseudo'], default='full')
    args = parser.parse_args()
    column, length = ('hla_seq', 182) if args.hla == 'full' else ('hla_pseudoseq', 34)
    out = OUT if args.hla == 'full' else OUT / 'pseudosequence'
    started = time.perf_counter()
    data = load_data()
    assert not data.duplicated(['allele', 'peptide']).any()
    assert data.peptide.str.len().eq(9).all() and data[column].str.len().eq(length).all()
    assert data.groupby('allele')[column].nunique().eq(1).all()
    names = sorted(data.allele.unique())
    allele_id = pd.Categorical(data.allele, categories=names).codes
    hlas = data.drop_duplicates('allele').set_index('allele').loc[names, column]
    unique_peptides, peptide_id = np.unique(data.peptide.to_numpy(), return_inverse=True)
    peptide_distance = hamming_matrix(encode(unique_peptides))
    hla_distance = hamming_matrix(encode(hlas))
    values = data.thalf_hours.to_numpy()
    ids = data.row_id.to_numpy()
    prediction = {method: np.full(len(data), np.nan) for method in METHODS}
    details = []
    for a, name in enumerate(names):
        train = np.flatnonzero(allele_id != a)
        test = np.flatnonzero(allele_id == a)
        hdist = hla_distance[a, allele_id[train]].astype(np.uint16)
        for offset in range(0, len(test), 128):
            query = test[offset:offset + 128]
            pdist = peptide_distance[peptide_id[query, None], peptide_id[train][None, :]].astype(np.uint16)
            for mode in ['raw', 'balanced']:
                # Multiplying normalized distance by HLA length * 9 preserves its minimum.
                distance = pdist + hdist if mode == 'raw' else length*pdist + 9*hdist
                nearest = distance.argmin(1)
                minimum = distance[np.arange(len(query)), nearest]
                tied = distance == minimum[:, None]
                tie_count = tied.sum(1)
                copied = values[train[nearest]]
                averaged = (tied * values[train][None, :]).sum(1) / tie_count
                prediction[mode + '_1nn'][query] = copied
                prediction[mode + '_tie_mean'][query] = averaged
                for j, row in enumerate(query):
                    chosen = train[nearest[j]]
                    assert allele_id[chosen] != a and chosen != row
                    details.append({
                        'row_id': int(ids[row]), 'allele': name, 'mode': mode,
                        'nearest_row_id': int(ids[chosen]), 'nearest_allele': data.allele.iloc[chosen],
                        'hla_hamming': int(hdist[nearest[j]]),
                        'peptide_hamming': int(pdist[j, nearest[j]]),
                        'distance': float(minimum[j] if mode == 'raw' else minimum[j]/(length*9)),
                        'equally_nearest_rows': int(tie_count[j]),
                        'prediction_1nn': float(copied[j]),
                        'prediction_tie_mean': float(averaged[j]),
                    })
        if (a + 1) % 15 == 0:
            print(f'{a + 1}/75 held-out alleles; {time.perf_counter()-started:.1f}s', flush=True)
    for p in prediction.values():
        assert np.isfinite(p).all()
    neighbours = pd.DataFrame(details)
    assert len(neighbours) == 2 * len(data)
    assert (neighbours.allele != neighbours.nearest_allele).all()
    assert (neighbours.row_id != neighbours.nearest_row_id).all()
    records = []
    for method, p in prediction.items():
        for subset in ['all', 'nonzero']:
            for a, name in enumerate(names):
                use = allele_id == a
                if subset == 'nonzero':
                    use &= values > 0
                y, pred = values[use], p[use]
                constant = np.unique(pred).size < 2
                rho = 0. if constant else float(spearmanr(y, pred).statistic)
                records.append({'method': method, 'subset': subset, 'allele': name,
                                'spearman': rho, 'n': len(y), 'constant_prediction': constant})
    per = pd.DataFrame(records)
    assert per.spearman.between(-1, 1).all()
    summary = per.groupby(['method', 'subset']).spearman.mean().unstack()
    summary = summary.loc[list(METHODS)].reset_index()
    summary['model'] = summary.method.map(METHODS)
    reference = pd.read_csv(ROOT / 'outputs/followup/pseudoseq_if/comparison.csv')
    comparison = pd.concat([
        reference[['model', 'spearman_nonzero', 'spearman_all']],
        summary.rename(columns={'nonzero': 'spearman_nonzero', 'all': 'spearman_all'})[
            ['model', 'spearman_nonzero', 'spearman_all']],
    ], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    neighbours.to_csv(out / 'neighbors.csv', index=False)
    per.to_csv(out / 'per_allele_metrics.csv', index=False)
    summary.to_csv(out / 'scores.csv', index=False)
    comparison.to_csv(out / 'comparison.csv', index=False)
    diagnostics = neighbours.groupby('mode').agg(
        mean_hla_hamming=('hla_hamming', 'mean'),
        mean_peptide_hamming=('peptide_hamming', 'mean'),
        same_peptide_fraction=('peptide_hamming', lambda x: float((x == 0).mean())),
        same_hla_sequence_fraction=('hla_hamming', lambda x: float((x == 0).mean())),
        tied_query_fraction=('equally_nearest_rows', lambda x: float((x > 1).mean())),
        mean_tied_rows=('equally_nearest_rows', 'mean')).reset_index()
    diagnostics.to_csv(out / 'neighbor_diagnostics.csv', index=False)
    protocol = {
        'split': '75-fold leave-one-allele-out; every candidate row from another allele',
        'rows': len(data), 'unique_peptides': len(unique_peptides),
        'hla_representation': args.hla, 'hla_length': length,
        'raw_distance': f'Hamming({length}-residue {column}) + Hamming(9-residue peptide)',
        'balanced_distance': f'Hamming(HLA)/{length} + Hamming(peptide)/9',
        'one_neighbor_tie_rule': 'First training row in original dataset order; independent of target',
        'tie_mean_rule': 'Arithmetic mean stability in hours across every equally nearest training row',
        'training_zeros_retained': True, 'nonzero_filter': 'Test metric only',
        'metric': 'Mean within-allele Spearman; constant prediction gets zero, matching existing reports',
        'no_tuning_or_model_training': True,
        'exact_test_coverage': True, 'no_heldout_allele_neighbors': True,
        'elapsed_seconds': time.perf_counter()-started,
        'memory_note': 'Peptide distance cache uses about 32 MB; candidate distances are computed in 128-row batches.',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    html = '''<!doctype html><meta charset="utf-8"><title>Naive nearest-neighbour baseline</title>
<style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#183144}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;border-bottom:1px solid #ccd8df;text-align:left}</style>
<h1>Naive nearest-neighbour stability transfer</h1><p>The same 75-fold leave-one-allele-out evaluation as the neural-network comparison. Each query can use only rows from the other 74 alleles. No model training or distance tuning.</p>
<p>Raw distance is HLA Hamming + peptide Hamming. The predefined balanced alternative divides by the HLA input length and 9 respectively. Strict 1NN copies the stability of the first closest training row; tie averaging uses the mean of all equally closest rows to remove arbitrary row-order dependence.</p>'''
    html += comparison.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)
    html += '<p>Scores are average within-allele Spearman. Neural-network rows average five seed-specific scores; nearest neighbour is deterministic. Nonzero filtering is applied only for test metrics; zero-valued training labels remain eligible.</p><h2>Which neighbours were selected?</h2>'
    html += diagnostics.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)
    html += f'<p>Completed in {protocol["elapsed_seconds"]:.1f} seconds locally. Shared peptides across alleles are allowed in this evaluation; the exact-peptide fraction above quantifies this source of label transfer. This baseline does not isolate a foundation-model mechanism.</p>'
    html += '<p><a href="comparison.csv">Scores CSV</a> · <a href="neighbors.csv">Every selected neighbour</a> · <a href="protocol.json">Protocol and checks</a></p>'
    (out / 'report.html').write_text(html)
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(diagnostics.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(json.dumps(protocol, indent=2))
    update_combined_table()


def update_combined_table():
    """Reuse saved results for both HLA representations; no repeated neighbour searches."""
    reference = pd.read_csv(ROOT / 'outputs/followup/pseudoseq_if/comparison.csv')
    tables = [reference[['model', 'spearman_nonzero', 'spearman_all']]]
    descriptions = {
        'raw_1nn': '1NN', 'raw_tie_mean': 'tied mean',
        'balanced_1nn': 'normalized 1NN', 'balanced_tie_mean': 'normalized tied mean',
    }
    for label, folder in [('Full HLA + peptide', OUT),
                           ('Pseudosequence + peptide', OUT / 'pseudosequence')]:
        if not (folder / 'scores.csv').exists():
            continue
        saved = pd.read_csv(folder / 'scores.csv')
        saved['model'] = saved.method.map(lambda m: f'{label}, {descriptions[m]}')
        tables.append(saved.rename(columns={'nonzero': 'spearman_nonzero', 'all': 'spearman_all'})[
            ['model', 'spearman_nonzero', 'spearman_all']])
    comparison = pd.concat(tables, ignore_index=True)
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    html = '''<!doctype html><meta charset="utf-8"><title>Nearest-neighbour comparison</title>
<style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#183144}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;border-bottom:1px solid #ccd8df;text-align:left}</style>
<h1>Nearest-neighbour and neural-model comparison</h1><p>The same 75-fold leave-one-allele-out evaluation for every row. Nearest neighbours always come from another allele. Scores are mean within-allele Spearman, with nonzero and all-half-life metrics separately. Neural-model scores average five seeds; nearest-neighbour models are deterministic.</p>'''
    html += comparison.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)
    html += '<p>Full HLA distance uses 182 residues; pseudosequence distance uses 34 selected residues. Raw distance sums HLA and nine-residue peptide Hamming counts. Normalized distance divides each count by its sequence length. 1NN copies one nearest stability value, taking the first dataset row if tied. Tied mean averages the stability of all equally nearest rows.</p>'
    html += '<p>Only the requested pseudosequence lookup was added; full-HLA and neural results are reused. Neither distance weighting nor tie rule was tuned against test labels.</p>'
    html += '<p><a href="comparison.csv">Combined CSV table</a> · <a href="neighbors.csv">Full-HLA neighbours</a> · <a href="pseudosequence/neighbors.csv">Pseudosequence neighbours</a> · <a href="pseudosequence/protocol.json">Pseudosequence protocol</a></p>'
    (OUT / 'report.html').write_text(html)
    print('Combined comparison:')
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


if __name__ == '__main__':
    main()
