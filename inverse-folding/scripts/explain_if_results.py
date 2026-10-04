"""Saved-output diagnostics and architecture report; no fitting or IF inference."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import followup_train as ft
from compare_pseudoseq_if import OUT, ARM, LABELS
from splits import load_data


def correlation(y, p):
    if np.unique(p).size < 2:
        return 0.
    return float(spearmanr(y, p).statistic)


def effects():
    per = pd.read_csv(OUT / 'comparison_per_allele_metrics.csv')
    q = per[(per.metric == 'spearman') & (per.subset == 'nonzero') &
            per.arm.isin(['pseudo', ARM])]
    paired = q.pivot(index=['allele', 'seed'], columns='arm', values='value')
    paired['delta'] = paired[ARM] - paired.pseudo
    paired.to_csv(OUT / 'diagnostic_seed_effects.csv')
    by = paired.groupby('allele').agg(
        baseline=('pseudo', 'mean'), augmented=(ARM, 'mean'),
        delta=('delta', 'mean'), seed_delta_sd=('delta', 'std'),
        seed_min=('delta', 'min'), seed_max=('delta', 'max'),
        seeds_losing=('delta', lambda x: int((x < 0).sum())))
    context = pd.read_csv(ft.ROOT / 'outputs/followup/allele_context.csv').set_index('allele')
    z = pd.read_csv(ft.ROOT / 'outputs/followup/zeroshot_per_allele.csv')
    z = z[(z.arm == 'pretrained') & (z.subset == 'nonzero') & (z.metric == 'spearman')]
    by = by.join(context).join(z.set_index('allele').value.rename('standalone_if_rho'))
    by['locus'] = by.index.str.extract(r'HLA-([AB])', expand=False)
    by.sort_values('delta').to_csv(OUT / 'allele_diagnostics.csv')
    losers = by[by.delta < 0].sort_values('delta')
    losers.to_csv(OUT / 'worsening_alleles.csv')
    groups = pd.read_csv(ft.ROOT / 'outputs/followup/splits/allele_groups.csv').set_index('allele')
    clustered = by.join(groups[['cluster']]).groupby('cluster').delta.agg(['sum', 'count'])
    rng = np.random.default_rng(20261004)
    sampled = rng.integers(0, len(clustered), size=(10000, len(clustered)))
    boot = clustered['sum'].to_numpy()[sampled].sum(1) / clustered['count'].to_numpy()[sampled].sum(1)
    associations = []
    for column in ['n_nonzero', 'zero_fraction', 'nearest_train_full_hamming',
                   'nearest_template_full_hamming', 'loao_fraction_peptides_seen_in_train',
                   'standalone_if_rho']:
        associations.append({'candidate_explanation': column,
                             'spearman_with_gain': correlation(by[column], by.delta),
                             'interpretation': 'Exploratory association; not a causal test.'})
    pd.DataFrame(associations).to_csv(OUT / 'diagnostic_associations.csv', index=False)
    locus = by.groupby('locus').agg(n_alleles=('delta', 'size'),
                                   mean_gain=('delta', 'mean'), median_gain=('delta', 'median'))
    locus.to_csv(OUT / 'locus_effects.csv')
    summary = {
        'mean_gain': float(by.delta.mean()), 'median_gain': float(by.delta.median()),
        'improved': int((by.delta > 0).sum()), 'worsened': len(losers),
        'losing_in_all_five_seeds': int((losers.seeds_losing == 5).sum()),
        'losing_in_at_least_four_seeds': int((losers.seeds_losing >= 4).sum()),
        'losing_only_two_or_three_seeds': int(losers.seeds_losing.isin([2, 3]).sum()),
        'cluster_bootstrap_groups': len(clustered),
        'cluster_bootstrap_95_interval': np.quantile(boot, [.025, .975]).tolist(),
        'mean_gain_without_five_largest_gains': float(by.delta.sort_values().iloc[:-5].mean()),
        'mean_gain_without_worst_loss': float(by.delta.sort_values().iloc[1:].mean()),
        'locus_effects': locus.reset_index().to_dict('records'),
        'associations': associations,
        'interval_note': 'Descriptive resampling of whole HLA sequence groups, preserving the allele-weighted mean. It addresses one source of relatedness but not all dependence or external validity.',
    }
    return by, losers, summary


def ensemble_saved_predictions():
    data = load_data()
    metrics = []
    predictions = []
    for i, (allele, rows) in enumerate(data.groupby('allele', sort=True)):
        ids = rows.row_id.to_numpy()
        y = rows.thalf_hours.to_numpy()
        scores = {}
        for arm in LABELS:
            arrays = []
            for seed in ft.HEAD_SEEDS:
                meta = {'split': 'leave_one_allele_out', 'fold': allele, 'axis': 'full',
                        'fraction': 1., 'arm': arm, 'seed': seed,
                        'schedule': 'fixed20', 'loss': 'ranking'}
                folder = OUT if arm == ARM else ft.OUT / ('main_0' if i < 38 else 'main_1')
                path = folder / 'runs' / (ft.identity(meta) + '.npz')
                with np.load(path) as saved:
                    np.testing.assert_array_equal(ids, saved['row_id'])
                    arrays.append(saved['prediction'])
            # Rank all candidate peptides before any target-based subset filtering.
            # The combination uses no labels or fitted blending weights.
            scores[arm] = (rankdata(np.stack(arrays), axis=1) / len(ids)).mean(0)
        scores['fixed_half_blend'] = .5 * scores['pseudo'] + .5 * scores[ARM]
        for arm, p in scores.items():
            predictions.extend({'row_id': int(r), 'allele': allele, 'model': arm,
                                'rank_ensemble_score': float(s)} for r, s in zip(ids, p))
            for subset in ['all', 'nonzero']:
                keep = np.ones(len(y), bool) if subset == 'all' else y > 0
                metrics.append({'model': arm, 'allele': allele, 'subset': subset,
                                'spearman': correlation(y[keep], p[keep]), 'n': int(keep.sum())})
    per = pd.DataFrame(metrics)
    per.to_csv(OUT / 'ensemble_per_allele.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'ensemble_predictions.csv', index=False)
    summary = per.groupby(['model', 'subset']).spearman.mean().unstack()
    summary['label'] = [LABELS.get(a, '50/50 blend of pseudosequence and augmented rank ensembles')
                        for a in summary.index]
    summary.to_csv(OUT / 'ensemble_comparison.csv')
    return summary


def figures(by, losers):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bins = np.linspace(-.5, .3, 23)
    axes[0].hist([by.loc[by.delta < 0, 'delta'], by.loc[by.delta >= 0, 'delta']],
                 bins=bins, stacked=True, color=['#bd553e', '#207e91'])
    axes[0].axvline(0, color='#334155', lw=1)
    axes[0].axvline(by.delta.mean(), color='#207e91', ls='--', label=f'Mean +{by.delta.mean():.3f}')
    axes[0].set(xlabel='Gain in within-allele Spearman', ylabel='Number of alleles',
                title='54 improve; 21 worsen')
    axes[0].legend(frameon=False)
    colors = np.where(by.delta < 0, '#bd553e', '#207e91')
    axes[1].scatter(by.n_nonzero, by.delta, c=colors, alpha=.8, s=32)
    axes[1].axhline(0, color='#334155', lw=1)
    axes[1].set(xscale='log', xlabel='Nonzero measurements for held-out allele',
                ylabel='Gain in within-allele Spearman', title='Losses occur with small and large samples')
    for allele in ['HLA-A*68:02', 'HLA-A*25:01']:
        row = by.loc[allele]
        axes[1].annotate(allele.replace('HLA-', ''), (row.n_nonzero, row.delta),
                         xytext=(6, 5), textcoords='offset points', fontsize=9)
    fig.savefig(OUT / 'allele_effects.png', dpi=170)
    fig.savefig(OUT / 'allele_effects.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ypos = np.arange(len(losers))
    ax.errorbar(losers.delta, ypos,
                xerr=np.stack([losers.delta-losers.seed_min, losers.seed_max-losers.delta]),
                fmt='o', color='#bd553e', ecolor='#c8a497', capsize=3)
    ax.axvline(0, color='#334155', lw=1)
    labels = [f'{a.replace("HLA-", "")}  (n={int(r.n_nonzero)}; losses {int(r.seeds_losing)}/5 seeds)'
              for a, r in losers.iterrows()]
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set(xlabel='Gain in within-allele Spearman',
           title='The 21 seed-averaged losses\nWhiskers show the five-seed range, not confidence intervals')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(OUT / 'worsening_alleles.png', dpi=170)
    fig.savefig(OUT / 'worsening_alleles.pdf')
    plt.close(fig)


def report(by, losers, summary, ensemble):
    ci = summary['cluster_bootstrap_95_interval']
    source = pd.read_csv(OUT / 'comparison.csv').set_index('arm')
    en = ensemble.loc[list(LABELS)].copy()
    en['mean_single_seed_nonzero'] = source.spearman_nonzero
    en = en[['label', 'mean_single_seed_nonzero', 'nonzero', 'all']]
    en.columns = ['Model', 'Original mean seed score', 'Rank ensemble, nonzero', 'Rank ensemble, all']
    table = losers.reset_index()[['allele', 'n_nonzero', 'baseline', 'augmented',
                                  'delta', 'seeds_losing', 'seed_min', 'seed_max']]
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What the inverse-folding gain means</title><style>
body{font:17px/1.6 system-ui;color:#173040;max-width:1200px;margin:36px auto;padding:0 24px}
h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border-bottom:1px solid #ccd8df;text-align:left}
img{width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:14px;color:#546974}.note{background:#edf5f7;padding:18px;border-radius:10px}.scroll{overflow-x:auto}a{color:#126b81}
</style><h1>What the inverse-folding gain means</h1>
<p>This extension reads existing predictions and cached metrics. It performs no training, foundation-model inference, or parameter tuning.</p>
<h2>The four architectures</h2><figure><img src="architectures.svg" alt="Four models share the same small neural-network head, with different active HLA and inverse-folding inputs."></figure>
<p>Yes: pseudosequence + inverse folding uses the same small supervised network as the baseline. Its two hidden layers contain 64 and 32 ReLU units, followed by one linear ranking score. There is no foundation-model fine-tuning. All models have the same nominal 3,870-input allocation, matched initial weights, data, folds, five seeds, ranking loss, 20 epochs, learning rate 0.001 and batch size 4,096. Masked and zero channels make active capacity differ. The common allocation is 249,857 parameters; removing identically zero coordinates leaves 36,673 stored parameters per head.</p>
<p>The nine observed-residue log probabilities supply peptide–HLA interactions directly: log p(actual residue at position i | HLA, template geometry). The baseline must learn these interactions from separate peptide and HLA encodings. The 40 P2/P9 preference probabilities are constant within each allele, so they influence within-allele ranking only through interaction with peptide-dependent inputs. The mean log probability is redundant with the nine positional values.</p>
<h2>How substantial is the gain?</h2>'''
    page += (f'<p>The gain is <strong>+{summary["mean_gain"]:.4f}, not +0.3</strong>: 0.4114 to 0.4449 '
             'mean within-allele Spearman on nonzero half-lives. All five seeds improve. The original descriptive '
             'allele-bootstrap interval is +0.0114 to +0.0532. '
             f'Resampling all {summary["cluster_bootstrap_groups"]} whole HLA sequence groups gives a descriptive interval '
             f'of {ci[0]:+.4f} to {ci[1]:+.4f}. This addresses some allele relatedness, while overlapping training folds '
             'and reuse of one assay cohort still limit formal statistical claims.</p>')
    page += (f'<p>The median allele gain is {summary["median_gain"]:+.4f}. Removing the five largest improvements '
             f'leaves a mean gain of {summary["mean_gain_without_five_largest_gains"]:+.4f}. '
             'These are sensitivity descriptions, not additional independent confirmations.</p>')
    page += '<h2>Why did 21 alleles get worse?</h2>'
    page += (f'<p>Of the 21 seed-averaged losses, <strong>{summary["losing_in_all_five_seeds"]} lose in every seed</strong>, '
             f'{summary["losing_in_at_least_four_seeds"]} lose in at least four, and '
             f'{summary["losing_only_two_or_three_seeds"]} lose in only two or three. These counts describe training consistency on the same test peptides; they are not confidence in a new biological sample.</p>')
    page += '<p>The worst loss, A*68:02, has only 16 nonzero measurements. A*69:01 has 10 and substantial seed variability. However, A*25:01, A*26:02, A*30:01, B*35:03 and B*81:01 have hundreds of measurements and repeated losses, so small test samples cannot explain everything.</p>'
    page += '<figure><img src="allele_effects.png" alt="Distribution of allele gains and their relation to test sample size."></figure><figure><img src="worsening_alleles.png" alt="All 21 worsening alleles, with seed ranges and sample sizes."></figure>'
    page += '<p>HLA locus shows heterogeneity, but is exploratory and affected by individual alleles:</p>'
    page += pd.DataFrame(summary['locus_effects']).to_html(index=False, float_format=lambda x:f'{x:.4f}', border=0)
    page += '<p>Simple associations with sample count, nearest training/template sequence distance, and standalone IF ranking do not establish an explanation. Shared geometry, imperfect motifs, the mismatch between sequence compatibility and dissociation kinetics, or undue reliance on the auxiliary features remain hypotheses. Entropy or template spread is not a calibrated reliability measure.</p>'
    page += '<div class="scroll">' + table.to_html(index=False, float_format=lambda x:f'{x:.4f}', border=0) + '</div>'
    page += '<h2>A performance check that uses no new training</h2><p>For every model, average the within-allele percentile ranks from its five saved heads. All candidate peptides are ranked before filtering by measured target. This uses no labels to set weights. The ensemble is different from the original average of five seed-specific correlations and should be reported separately:</p>'
    page += '<div class="scroll">' + en.to_html(index=False, float_format=lambda x:f'{x:.4f}', border=0) + '</div>'
    blend = ensemble.loc['fixed_half_blend', 'nonzero']
    page += f'<p>A fixed 50/50 blend of the pseudosequence and augmented rank ensembles scores {blend:.4f} on nonzero rows. This is an exploratory robustness check, not a validation-selected best model. No blend weight was tuned against the test labels.</p>'
    page += '<h2>Three small next experiments, if desired</h2><ol><li><strong>Cached random-feature control:</strong> pseudosequence + random LigandMPNN features, with the same head and budget. This is the most direct remaining test of whether pretrained weights add benefit beyond another feature transformation.</li><li><strong>A safer auxiliary branch:</strong> a regularized residual correction to the baseline, with its strength selected on inner training-allele validation. This may retain baseline behaviour where IF is unhelpful. The 21 outer-test losers must not be used to choose the correction.</li><li><strong>A small feature ablation:</strong> nine observed-residue log probabilities versus the 40 anchor preferences. This tests whether peptide-specific compatibility or the HLA motif representation carries the gain. The mean score adds no information beyond the nine values.</li></ol>'
    page += '<p>These are proposals, not completed experiments. The current positive claim is that frozen LigandMPNN features improve the strongest tested sequence-only baseline at this fixed budget. They do not establish a specific steric mechanism or superiority after fully optimizing the baseline.</p><p><a href="comparison.csv">Original four-model table</a> · <a href="allele_diagnostics.csv">Allele diagnostics</a> · <a href="ensemble_comparison.csv">Ensemble scores</a> · <a href="architectures.svg">Download architecture diagram</a></p></html>'
    (OUT / 'diagnostics.html').write_text(page)


def main():
    by, losers, summary = effects()
    ensemble = ensemble_saved_predictions()
    figures(by, losers)
    summary['no_new_training_or_inference'] = True
    summary['ensemble_note'] = 'Label-free mean percentile rank across five saved seeds, computed on all candidate peptides before subset evaluation.'
    (OUT / 'diagnostics_summary.json').write_text(json.dumps(summary, indent=2))
    report(by, losers, summary, ensemble)
    print(json.dumps(summary, indent=2))
    print(ensemble.to_string(float_format=lambda x:f'{x:.6f}'))


if __name__ == '__main__':
    main()
