"""Complete only the missing pseudosequence + frozen IF arm of fixed20 LOAO.

Reuse followup_train unchanged: its 'pseudo' head supplies the 34-position mask,
while its auxiliary input is populated with cached pretrained IF features.
Saved results use the distinct public arm name 'pseudo_pretrained'.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import followup_train as ft
from followup_aggregate import aggregate, paired_comparisons

OUT = ft.ROOT / 'outputs/followup/pseudoseq_if'
ARM = 'pseudo_pretrained'
BASE_ARMS = ['pseudo', 'full', 'pretrained']
LABELS = {
    'pseudo': 'Pseudosequence',
    'full': 'Full HLA sequence',
    'pretrained': 'Full HLA sequence + inverse folding',
    ARM: 'Pseudosequence + inverse folding',
}


def prepare(out):
    data = ft.Data(['pretrained'])
    # 'pseudo' selects the original HLA mask, not a different numerical kernel.
    data.aux['pseudo'] = data.aux['pretrained']
    kernel = {
        'training_implementation': {
            f.__qualname__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in [ft.PackedHead, ft.make_schedule, ft.fit, ft.predict,
                      ft.Data.standardize, ft.representations]},
        'data_sha256': ft.sha(ft.ROOT / 'resources/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv'),
        'batch_size': 4096, 'torch': ft.torch.__version__, 'protocol_version': 1,
    }
    data.kernel_fingerprint = ft.identity(kernel)
    data.batch_size = 4096
    nonzero_hash = hashlib.sha256(data.aux['pretrained'].tobytes()).hexdigest()
    zero_hash = hashlib.sha256(np.zeros_like(data.aux['pretrained']).tobytes()).hexdigest()
    data.feature_fingerprints = {'pseudo': zero_hash, 'full': zero_hash,
                                'pretrained': nonzero_hash, ARM: nonzero_hash}
    for shard in ['main_0', 'main_1']:
        base = ft.OUT / shard
        assert json.loads((base / 'kernel_protocol.json').read_text()) == {
            'fingerprint': data.kernel_fingerprint, 'kernel': kernel}
        assert json.loads((base / 'context_protocol.json').read_text()) == ft.context_protocol()
        protocols = [json.loads(p.read_text()) for p in base.glob('provenance_*.json')]
        assert any(p['source_sha256'] == ft.sha(ft.__file__) and
                   p['sources']['pretrained'] == data.sources['pretrained'] and
                   p['seeds'] == ft.HEAD_SEEDS for p in protocols)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'arm': ARM, 'internal_head_selector': 'pseudo',
        'auxiliary': 'The same 50 cached pretrained features as the full-HLA augmented arm.',
        'feature_source': data.sources['pretrained'],
        'feature_array_sha256': nonzero_hash,
        'kernel': kernel, 'kernel_fingerprint': data.kernel_fingerprint,
        'training_source_sha256': ft.sha(ft.__file__),
        'context': ft.context_protocol(), 'seeds': ft.HEAD_SEEDS,
        'split': 'leave_one_allele_out', 'folds': 75, 'loss': 'ranking',
        'epochs': 20, 'learning_rate': .001, 'batch_size': 4096, 'threads': 2,
        'initialization_and_pair_schedule': 'Unchanged original stable_seed and training functions.',
        'no_selection': True, 'no_feature_extraction': True,
        'baseline_metrics_sha256': ft.sha(ft.ROOT / 'outputs/followup/complete_per_allele_metrics.csv'),
    }
    path = out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Incompatible saved comparison protocol'
    ft.atomic_json(path, protocol)
    ft.ensure_context_guard(out)
    return data, ft.loao_cells(data)


def run_path(cell, arm, seed, folder):
    meta = {k: cell[k] for k in ['split', 'fold', 'axis', 'fraction']}
    meta.update(arm=arm, seed=seed, schedule='fixed20', loss='ranking')
    return folder / 'runs' / ft.identity(meta)


def train(data, cells, out):
    start = time.perf_counter()
    completed = 0
    for i, cell in enumerate(cells):
        train = data.positions(cell['final_train_ids'])
        test = data.positions(cell['test_ids'])
        assert not set(train) & set(test)
        for seed in ft.HEAD_SEEDS:
            if ft.already_done(out, data, cell, ARM, seed, 'fixed20', 'ranking'):
                continue
            head_seed = ft.stable_seed(seed, cell['split'], cell['fold'])
            predictions, stats = ft.fit(data, ['pseudo'], train, test, head_seed,
                                        .001, 20, batch_size=4096)
            saved = {k: v for k, v in stats.items()
                     if k not in ['scaling', 'last_losses', 'snapshots']}
            saved.update(epochs=20, learning_rate=.001, head_seed=head_seed,
                         scaling=stats['scaling']['pseudo'], packed_arms=[ARM],
                         internal_head_selector='pseudo',
                         last_batch_loss=stats['snapshots'][20]['last_losses']['pseudo'],
                         seconds_to_checkpoint=stats['snapshots'][20]['seconds'])
            ft.save_result(out, data, cell, ARM, seed, 'fixed20', 'ranking', test,
                           predictions[20][0], saved)
            completed += 1
        elapsed = time.perf_counter() - start
        ft.atomic_json(out / 'progress.json', {
            'completed_folds': i + 1, 'total_folds': 75, 'new_fits_this_invocation': completed,
            'elapsed_seconds': elapsed, 'complete': i + 1 == 75})
        print(json.dumps({'fold': i + 1, 'of': 75, 'allele': cell['fold'],
                          'new_fits': completed, 'seconds': round(elapsed, 1)}), flush=True)
    ft.consolidate(out, data.kernel_fingerprint)


def summarize(data, cells, out):
    ft.consolidate(out, data.kernel_fingerprint)
    original = pd.read_csv(ft.ROOT / 'outputs/followup/complete_per_allele_metrics.csv')
    original = original[(original.split == 'leave_one_allele_out') &
                        (original.loss == 'ranking') & (original.schedule == 'fixed20') &
                        original.arm.isin(BASE_ARMS)]
    added = pd.read_csv(out / 'per_allele_metrics.csv')
    per = pd.concat([original, added], ignore_index=True)
    assert set(per.arm) == set(BASE_ARMS + [ARM])
    assert len(per) == 4 * 75 * 5 * 6
    assert per.groupby(['arm', 'seed', 'subset', 'metric']).allele.nunique().eq(75).all()
    assert set(per.seed) == set(ft.HEAD_SEEDS)
    assert np.isfinite(per.value).all()
    # Verify every saved prediction and recompute the headline metric independently.
    checked, counts = 0, {a: {s: 0 for s in ft.HEAD_SEEDS} for a in LABELS}
    metric_index = per.set_index(['arm', 'seed', 'allele', 'subset', 'metric'])
    for i, cell in enumerate(cells):
        test = data.positions(cell['test_ids'])
        for seed in ft.HEAD_SEEDS:
            for arm in LABELS:
                folder = out if arm == ARM else ft.OUT / ('main_0' if i < 38 else 'main_1')
                path = run_path(cell, arm, seed, folder)
                result = json.loads(path.with_suffix('.json').read_text())
                assert result['fingerprint'] == ft.result_fingerprint(data, cell, arm, 'ranking', 'fixed20')
                assert result['training']['head_seed'] == ft.stable_seed(seed, cell['split'], cell['fold'])
                assert result['training']['epochs'] == 20
                assert result['training']['learning_rate'] == .001
                assert result['training']['optimizer_steps'] == 20 * math.ceil(len(cell['final_train_ids']) / 4096)
                with np.load(path.with_suffix('.npz')) as z:
                    np.testing.assert_array_equal(z['row_id'], data.frame.row_id.to_numpy()[test])
                    pred = z['prediction']
                    assert np.isfinite(pred).all()
                for subset in ['all', 'nonzero']:
                    keep = np.ones(len(test), bool) if subset == 'all' else data.target[test] > 0
                    rho = float(spearmanr(data.target[test][keep], pred[keep]).statistic)
                    if np.unique(pred[keep]).size < 2:
                        rho = 0.
                    old = metric_index.loc[(arm, seed, cell['fold'], subset, 'spearman'), 'value']
                    np.testing.assert_allclose(rho, old, rtol=0, atol=1e-12)
                checked += 1
                counts[arm][seed] += len(test)
    assert all(n == len(data.frame) for arm in counts.values() for n in arm.values())
    per.to_csv(out / 'comparison_per_allele_metrics.csv', index=False)
    _, _, summary = aggregate(per, out)
    contrast, _ = paired_comparisons(per[per.metric == 'spearman'], [
        ('pseudo', ARM), ('full', 'pretrained'), ('pretrained', ARM),
        ('full', 'pseudo'), ('pseudo', 'pretrained')], out)
    table = []
    for arm, label in LABELS.items():
        q = summary[(summary.arm == arm) & (summary.metric == 'spearman') &
                    (summary.aggregation == 'macro')].set_index('subset')
        table.append({'model': label, 'arm': arm, 'spearman_nonzero': q.loc['nonzero', 'mean'],
                      'spearman_all': q.loc['all', 'mean'],
                      'seed_sd_nonzero': q.loc['nonzero', 'seed_sd'],
                      'seeds': 5, 'heldout_alleles': 75})
    table = pd.DataFrame(table)
    table.to_csv(out / 'comparison.csv', index=False)
    ft.atomic_json(out / 'integrity.json', {
        'passed': True, 'saved_predictions_checked': checked,
        'new_fits': 375, 'reused_fits': 1125, 'exact_test_coverage_per_arm_seed': True,
        'headline_metrics_independently_recomputed': True,
        'source_kernel_features_membership_initialization_budget_verified': True,
        'rows_per_arm_seed': len(data.frame),
    })
    shown = table[['model', 'spearman_nonzero', 'spearman_all']].rename(columns={
        'model': 'Model', 'spearman_nonzero': 'Spearman, nonzero', 'spearman_all': 'Spearman, all'})
    display_table = shown
    nearest_table = out / 'nearest_neighbor/comparison.csv'
    if nearest_table.exists():
        display_table = pd.read_csv(nearest_table).rename(columns={
            'model': 'Model', 'spearman_nonzero': 'Spearman, nonzero',
            'spearman_all': 'Spearman, all'})
    gains = contrast[contrast.subset == 'nonzero'][
        ['baseline', 'variant', 'delta', 'ci_low', 'ci_high', 'alleles_improved']].copy()
    gains['baseline'] = gains.baseline.map(LABELS)
    gains['variant'] = gains.variant.map(LABELS)
    report = '''<!doctype html><meta charset="utf-8"><title>Matched pseudosequence + inverse-folding comparison</title>
<style>body{font:17px/1.5 system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#172b38}
table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd6dd}</style>
<h1>Matched pseudosequence + inverse-folding comparison</h1>
<p>Average within-allele Spearman over 75 held-out alleles. The four neural models average five paired seeds
and share peptide inputs, 20 epochs, Adam learning rate 0.001, batch size 4,096, ranking loss,
and the original common head and initialization. Nearest-neighbour methods are deterministic label transfers
under the same held-out-allele split, with no model training.</p>'''
    report += display_table.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)
    if nearest_table.exists():
        report += '<p><a href="nearest_neighbor/report.html">Nearest-neighbour definitions and combined CSV table</a>. Full-HLA distances use 182 residues; pseudosequence distances use 34. Each also includes the nine-residue peptide.</p>'
    effect = contrast[(contrast.subset == 'nonzero') & (contrast.baseline == 'pseudo') &
                      (contrast.variant == ARM)].iloc[0]
    report += (f'<p><strong>Adding inverse-folding features to the pseudosequence baseline improves '
               f'nonzero Spearman by {effect.delta:.4f}</strong> '
               f'(descriptive 95% paired interval {effect.ci_low:.4f} to {effect.ci_high:.4f}), '
               f'with improvement on {int(effect.alleles_improved)} of 75 alleles after averaging seeds. '
               'Pseudosequence plus inverse-folding also outperforms full HLA plus inverse-folding '
               'under this matched training budget.</p>')
    report += '<p>Only pseudosequence + inverse-folding was newly trained (375 small-head fits). The other three models reuse 1,125 saved fits. The same cached 50 inverse-folding features were used; no new foundation-model inference, tuning, or external API calls.</p>'
    report += '<h2>Paired differences on nonzero half-lives</h2>'
    report += gains.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)
    report += '<p>Intervals are descriptive 95% allele-bootstrap intervals after averaging paired seed effects. Related alleles and overlapping training folds limit independence.</p>'
    report += '<p>This comparison measures the benefit of adding pretrained peptide–HLA compatibility features at the fixed training budget. Those features condition on the full HLA sequence and shared backbone templates even when the supervised head uses only the pseudosequence. Gains therefore do not isolate steric modelling or eliminate differences in active model capacity. No new random-feature or geometry controls were trained for the pseudosequence arm, and convergence was not established.</p>'
    report += '<p>A plausible explanation is that precomputed compatibility features make relevant peptide–HLA relationships easier for the small head to learn. Improvement when the head already receives the full HLA sequence shows that the benefit is not explained solely by providing omitted HLA positions. It does not prove a particular physical mechanism.</p>'
    report += '<p>Numerical reproducibility: the original two-thread CPU training is not bitwise deterministic; repeated single-fold baseline fits differed slightly in ranking. The new arm preserves the same source kernel, initializations, batch schedules, and five-seed protocol. See <a href="setup_checks.json">setup checks</a>, <a href="integrity.json">prediction validation</a>, and <a href="comparison.csv">the CSV table</a>.</p>'
    report += '<p>Reproduce or resume: <code>.venv/bin/python scripts/compare_pseudoseq_if.py</code>. Regenerate the table without training: append <code>--report-only</code>.</p>'
    if (out / 'architectures.svg').exists():
        report += '<h2>The four architectures</h2><img src="architectures.svg" style="width:100%;height:auto" alt="Four matched small neural-network heads with different active HLA and inverse-folding inputs">'
        report += '<p>All four use the same 64-unit and 32-unit hidden layers. Masked HLA positions and zero auxiliary channels make active capacity differ. Both inverse-folding arms receive indirect full-HLA context through the frozen feature extractor.</p>'
    if (out / 'diagnostics.html').exists():
        report += '<p><a href="diagnostics.html">Further analysis: the 21 worsening alleles, grouped uncertainty, and performance from combining saved seed predictions</a>. These diagnostics use saved outputs without new training.</p>'
    if (out / 'nearest_neighbor/distance_vs_spearman.png').exists():
        report += '<h2>Allele performance versus nearest-neighbour distance</h2><img src="nearest_neighbor/distance_vs_spearman.png" style="width:100%;height:auto" alt="75 nearest-neighbour points and 75 MHC baseline points plotted against mean nearest-pair Hamming distance">'
        report += '<p>The x-axis averages HLA + peptide Hamming distance to the nearest training pair over all rows for each allele. Both models use the same x coordinates. The y-axis is nonzero-half-life Spearman: strict 1NN versus the peptide + pseudosequence baseline averaged across five seeds.</p>'
        report += '<p><a href="nearest_neighbor/distance_vs_spearman.html">Explore allele tooltips</a> · <a href="nearest_neighbor/distance_vs_spearman.pdf">Download PDF</a> · <a href="nearest_neighbor/distance_vs_spearman.csv">Plot data</a></p>'
    if (out / 'nearest_neighbor/distance_vs_spearman_difference.png').exists():
        report += '<h2>Baseline minus nearest-neighbour performance</h2><img src="nearest_neighbor/distance_vs_spearman_difference.png" style="width:100%;height:auto" alt="75 allele differences in Spearman plotted against mean Hamming distance with a linear trendline">'
        report += '<p>Each point is baseline Spearman minus nearest-neighbour Spearman, at the same mean Hamming distance. Positive values favour the baseline. The trendline is ordinary least squares with equal weight per allele.</p>'
        report += '<p><a href="nearest_neighbor/distance_vs_spearman_difference.html">Explore allele tooltips</a> · <a href="nearest_neighbor/distance_vs_spearman_difference.pdf">Download PDF</a></p>'
    (out / 'report.html').write_text(report)
    print(shown.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(gains.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    ft.torch.set_num_threads(2)
    ft.torch.set_num_interop_threads(1)
    data, cells = prepare(OUT)
    if not args.report_only:
        train(data, cells, OUT)
    summarize(data, cells, OUT)


if __name__ == '__main__':
    main()
