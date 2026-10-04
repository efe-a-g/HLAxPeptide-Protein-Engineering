"""Plot saved allele-level performance against mean nearest-pair Hamming distance."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/followup/pseudoseq_if'
OUT = BASE / 'nearest_neighbor'
NS = 'http://www.w3.org/2000/svg'


def plot_difference(points):
    points = points.copy()
    points['spearman_difference'] = points.baseline_spearman - points.nn_spearman
    x, y = points.mean_hamming.to_numpy(), points.spearman_difference.to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(x.min(), x.max(), 200)
    stem = OUT / 'distance_vs_spearman_difference'
    points.to_csv(stem.with_suffix('.csv'))
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.20, top=.81)
    fig.suptitle('MHC baseline advantage over nearest neighbour',
                 x=.12, y=.96, ha='left', fontsize=15, fontweight='bold')
    fig.text(.12, .91, 'Leave-one-allele-out · 75 alleles · nonzero-half-life Spearman',
             fontsize=10.5, color='#526475')
    scatter = ax.scatter(x, y, s=52, color='#8256a7', alpha=.85, edgecolors='white',
                         linewidths=.6, label='Baseline Spearman − nearest-neighbour Spearman', zorder=3)
    scatter.set_gid('difference-points')
    ax.plot(fit_x, slope*fit_x + intercept, color='#42345c', linewidth=2,
            label='Linear trend: ordinary least squares, equal weight per allele', zorder=2)
    ax.axhline(0, color='#667787', linestyle='--', linewidth=1, zorder=0)
    ax.set_xlabel('Mean nearest-neighbour Hamming distance over all pairs for the allele\n'
                  '(182-residue HLA mismatches + 9-residue peptide mismatches)', labelpad=12)
    ax.set_ylabel('Spearman difference: baseline − nearest neighbour', labelpad=10)
    ax.grid(alpha=.15, zorder=0)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=10,
              borderaxespad=0)
    fig.text(.12, .025, 'Above zero: baseline performs better. Below zero: nearest neighbour performs better.\n'
                        'Baseline averages five seeds. The trendline is descriptive; each allele has equal weight.',
             fontsize=9, color='#526475', va='bottom')
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig)
    tree = ET.parse(stem.with_suffix('.svg'))
    group = tree.getroot().find(f'.//{{{NS}}}g[@id="difference-points"]')
    uses = group.findall(f'.//{{{NS}}}use')
    assert len(uses) == 75
    for point, (allele, row) in zip(uses, points.iterrows()):
        ET.SubElement(point, f'{{{NS}}}title').text = (
            f'{allele} | mean Hamming {row.mean_hamming:.3f} | '
            f'baseline − nearest neighbour {row.spearman_difference:+.3f}')
    tree.write(stem.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)
    svg = stem.with_suffix('.svg').read_text()
    svg = svg[svg.index('<svg'):]
    html = '<!doctype html><meta charset="utf-8"><title>Baseline minus nearest-neighbour Spearman</title><style>body{font:17px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#183144}svg{width:100%;height:auto}</style><p>Hover over a point to identify its allele.</p>'
    html += svg
    html += '<p>Each point is the purple baseline score minus the green nearest-neighbour score from the original chart. X is unchanged. The trendline is an unweighted ordinary-least-squares fit to the 75 allele points.</p>'
    html += '<p><a href="distance_vs_spearman_difference.pdf">PDF</a> · <a href="distance_vs_spearman_difference.csv">Data CSV</a></p>'
    stem.with_suffix('.html').write_text(html)
    stem.with_suffix('.json').write_text(json.dumps({
        'points': 75, 'difference': 'baseline_spearman - nn_spearman',
        'trendline': 'Unweighted ordinary least squares across alleles',
        'slope': float(slope), 'intercept': float(intercept),
        'r_squared': float(np.corrcoef(x, y)[0, 1]**2),
        'mean_difference': float(y.mean()),
    }, indent=2))
    print(f'Difference plot: 75 points; mean delta={y.mean():+.6f}; slope={slope:+.6f}.')


def main():
    neighbours = pd.read_csv(OUT / 'neighbors.csv')
    neighbours = neighbours[neighbours['mode'] == 'raw']
    assert not neighbours.row_id.duplicated().any()
    assert (neighbours.allele != neighbours.nearest_allele).all()
    np.testing.assert_array_equal(neighbours.distance,
                                  neighbours.hla_hamming + neighbours.peptide_hamming)
    x = neighbours.groupby('allele').agg(
        mean_hamming=('distance', 'mean'),
        mean_hla_hamming=('hla_hamming', 'mean'),
        mean_peptide_hamming=('peptide_hamming', 'mean'), n_all=('row_id', 'size'))
    nn = pd.read_csv(OUT / 'per_allele_metrics.csv')
    nn = nn[(nn.method == 'raw_1nn') & (nn.subset == 'nonzero')].set_index('allele')
    mhc = pd.read_csv(BASE / 'comparison_per_allele_metrics.csv')
    mhc = mhc[(mhc.arm == 'pseudo') & (mhc.subset == 'nonzero') &
              (mhc.metric == 'spearman')]
    assert mhc.groupby('allele').seed.nunique().eq(5).all()
    baseline = mhc.groupby('allele').value.mean().rename('baseline_spearman')
    points = x.join(nn[['spearman', 'n']].rename(columns={
        'spearman': 'nn_spearman', 'n': 'n_nonzero'})).join(baseline).sort_index()
    assert len(points) == 75 and not points.isna().any().any()
    np.testing.assert_allclose(points.mean_hamming,
                               points.mean_hla_hamming + points.mean_peptide_hamming)
    points.to_csv(OUT / 'distance_vs_spearman.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.subplots_adjust(left=.11, right=.97, bottom=.20, top=.81)
    fig.suptitle('Allele performance versus distance to the nearest training pair',
                 x=.11, y=.96, ha='left', fontsize=15, fontweight='bold')
    fig.text(.11, .91, 'Leave-one-allele-out · 75 alleles per model · nonzero-half-life Spearman',
             fontsize=10.5, color='#526475')
    ax.vlines(points.mean_hamming,
              np.minimum(points.nn_spearman, points.baseline_spearman),
              np.maximum(points.nn_spearman, points.baseline_spearman),
              colors='#b5c2cb', linewidth=.9, alpha=.60, zorder=1)
    nn_scatter = ax.scatter(points.mean_hamming, points.nn_spearman, s=49,
                            c='#087f8c', marker='o', alpha=.85, linewidths=.6,
                            edgecolors='white', label='Nearest neighbour: copy one closest row', zorder=3)
    baseline_scatter = ax.scatter(points.mean_hamming, points.baseline_spearman, s=49,
                                  c='#8256a7', marker='^', alpha=.85, linewidths=.6,
                                  edgecolors='white', label='MHC baseline: peptide + pseudosequence', zorder=3)
    nn_scatter.set_gid('nearest-neighbour-points')
    baseline_scatter.set_gid('baseline-points')
    ax.axhline(0, color='#667787', linestyle='--', linewidth=.9, zorder=0)
    ax.set_xlabel('Mean nearest-neighbour Hamming distance over all pairs for the allele\n'
                  '(182-residue HLA mismatches + 9-residue peptide mismatches)', labelpad=12)
    ax.set_ylabel('Within-allele Spearman', labelpad=10)
    ax.grid(alpha=.15, zorder=0)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=10,
              borderaxespad=0)
    fig.text(.11, .025, 'Each grey segment connects the same allele in both models. Baseline scores average five seeds.\n'
                        'Both colours use the same x coordinate; distance averages include zero-half-life rows.',
             fontsize=9, color='#526475', va='bottom')
    stem = OUT / 'distance_vs_spearman'
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig)
    # Native SVG tooltips identify every allele without cluttering the static chart.
    ET.register_namespace('', NS)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(stem.with_suffix('.svg'))
    for gid, column, model in [
        ('nearest-neighbour-points', 'nn_spearman', 'Nearest neighbour'),
        ('baseline-points', 'baseline_spearman', 'MHC baseline')]:
        group = tree.getroot().find(f'.//{{{NS}}}g[@id="{gid}"]')
        uses = group.findall(f'.//{{{NS}}}use')
        assert len(uses) == 75
        for point, (allele, row) in zip(uses, points.iterrows()):
            ET.SubElement(point, f'{{{NS}}}title').text = (
                f'{allele} | {model} | mean Hamming {row.mean_hamming:.3f} | '
                f'Spearman {row[column]:.3f} | {int(row.n_all)} total / '
                f'{int(row.n_nonzero)} nonzero measurements')
    tree.write(stem.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)
    svg = stem.with_suffix('.svg').read_text()
    svg = svg[svg.index('<svg'):]
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distance versus allele performance</title><style>
body{font:17px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#183144}
svg{width:100%;height:auto}a{color:#126b81}</style>
<p>Hover over a point to identify its allele, mean distance and score.</p>'''
    html += svg
    html += '<p>X is the mean distance to the nearest eligible training pair across <strong>all rows for each allele</strong>, using Hamming(HLA) + Hamming(peptide). The held-out allele is excluded from the candidate pool. Y is Spearman on nonzero half-lives. The MHC baseline is the peptide + 34-residue pseudosequence model, averaged over five seeds.</p>'
    html += '<p>There are 75 points per model and 150 points in total. Paired points share the same x coordinate; grey lines connect them. This is a descriptive comparison with training-data proximity.</p>'
    html += '<p><a href="distance_vs_spearman.png">PNG</a> · <a href="distance_vs_spearman.pdf">PDF</a> · <a href="distance_vs_spearman.svg">SVG</a> · <a href="distance_vs_spearman.csv">75-allele data table</a></p></html>'
    stem.with_suffix('.html').write_text(html)
    print(f'Wrote 75 allele pairs (150 points). Mean NN rho={points.nn_spearman.mean():.6f}; '
          f'mean baseline rho={points.baseline_spearman.mean():.6f}. No training.')
    plot_difference(points)


if __name__ == '__main__':
    main()
