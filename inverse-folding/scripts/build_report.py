"""Generate an offline HTML research report from saved, actual experiment outputs."""
from pathlib import Path
import base64
import html
import json
import os
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation import NOISE_CEILING, paired_allele_bootstrap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'outputs/figures'
COLORS = ['#177e89', '#c45d3e', '#6b5893', '#527a49', '#cfad49']

def model_label(value):
    aliases={'allele_mean':'Allele mean (global fallback)', 'blosum_pseudoseq':'BLOSUM + pseudosequence',
             'allele_embedding':'Learned allele embedding','pwm':'Allele PWM',
             'ligandmpnn_unconditional':'LigandMPNN · independent',
             'ligandmpnn_conditional':'LigandMPNN · autoregressive',
             'ligandmpnn_scrambled':'LigandMPNN · HLA shuffled'}
    return aliases.get(value,value.replace('baseline_plus_','Baseline + ').replace('unconditional','indep.').replace('conditional','AR').replace('logp_per_position','positions').replace('anchor_softmax','anchors').replace('mean_logp','scalar').replace('_',' '))

def figure(name, caption):
    path = Path(name) if Path(name).is_absolute() else FIG / name
    if not path.exists():
        return ''
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

def table(frame, digits=3):
    return frame.to_html(index=False, border=0, na_rep='NA', float_format=lambda x: f'{x:.{digits}f}', escape=True)

def finish_figure(fig, path, correlation=False):
    if correlation:
        fig.text(.5, .015, NOISE_CEILING, ha='center', fontsize=8, color='#555555')
        fig.tight_layout(rect=(0,.055,1,1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def model_figures(results, per):
    FIG.mkdir(parents=True, exist_ok=True)
    g = results[(results.split == 'leave_one_allele_out') & (results.aggregation == 'macro') &
                (results.metric == 'spearman') & results.value.notna()]
    if len(g):
        piv = g.pivot_table(index=['representation','loss'], columns='subset', values='value')
        fig, ax = plt.subplots(figsize=(10,max(4,len(piv)*.38+1)))
        y = np.arange(len(piv))
        for i, subset in enumerate(['all','nonzero']):
            if subset in piv:
                ax.barh(y+(i-.5)*.34,piv[subset],height=.32,label=subset,color=COLORS[i])
        ax.set_yticks(y, [f'{model_label(a)} · {b}' for a,b in piv.index],fontsize=8)
        ax.set_xlabel('Mean within-allele Spearman (75 held-out alleles; coverage in tables)')
        ax.set_title('Primary evaluation: leave one allele out')
        ax.axvline(0,c='#999999',lw=.7); ax.legend()
        finish_figure(fig, FIG/'model_loao.png',True)
    split_models=['allele_mean','pwm','allele_embedding','blosum_pseudoseq',
                  'baseline_plus_unconditional_all','baseline_plus_conditional_all']
    g = results[(results.aggregation == 'macro') & (results.metric == 'spearman') &
                results.loss.isin(['mse','reference']) & results.representation.isin(split_models)]
    if len(g):
        fig, axs = plt.subplots(1,2,figsize=(13,5),sharey=True)
        for ax,subset in zip(axs,['all','nonzero']):
            p = g[g.subset==subset].pivot(index='representation',columns='split',values='value').dropna(how='all')
            if len(p):
                p.index=[model_label(i) for i in p.index]
                p.plot.bar(ax=ax,color=COLORS,rot=40)
            ax.set_title(f'{subset} rows'); ax.set_ylabel('Mean within-allele Spearman'); ax.set_xlabel('')
            ax.tick_params(axis='x',labelsize=7)
        finish_figure(fig,FIG/'model_splits.png',True)
    lo = per[(per.split=='leave_one_allele_out') & (per.loss=='mse') & (per.subset=='nonzero') & (per.metric=='spearman')]
    baseline = 'blosum_pseudoseq'
    reps = sorted(set(lo.representation)-{baseline,'allele_mean','pwm','allele_embedding'})
    if baseline in set(lo.representation) and reps:
        p = lo.pivot(index='allele',columns='representation',values='value')
        ncols=min(3,len(reps));nrows=int(np.ceil(len(reps)/ncols))
        fig,axs=plt.subplots(nrows,ncols,figsize=(ncols*4.4,nrows*4.1),squeeze=False)
        for ax,rep in zip(axs.ravel(),reps):
            q=p[[baseline,rep]].dropna()
            ax.scatter(q[baseline],q[rep],c=COLORS[0],alpha=.65,s=22)
            ax.plot([-1,1],[-1,1],c='#777',lw=1,ls='--')
            short=model_label(rep)
            ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='Baseline allele Spearman',ylabel='Augmented allele Spearman',title=short)
            ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([-1,-.5,0,.5,1])
            ax.text(.04,.94,f'{len(q)} paired alleles',transform=ax.transAxes,va='top',fontsize=8)
        for ax in axs.ravel()[len(reps):]: ax.set_visible(False)
        finish_figure(fig,FIG/'model_paired_alleles.png',True)

def main():
    audit = json.loads((ROOT/'outputs/audit/audit.json').read_text())
    full = ROOT/'outputs/baselines/full'
    results = pd.read_csv(ROOT/'results.csv') if (ROOT/'results.csv').exists() else (pd.read_csv(full/'results.csv') if (full/'results.csv').exists() else pd.DataFrame())
    per = pd.read_csv(ROOT/'outputs/per_allele_metrics.csv') if (ROOT/'outputs/per_allele_metrics.csv').exists() else (pd.read_csv(full/'per_allele_metrics.csv') if (full/'per_allele_metrics.csv').exists() else pd.DataFrame())
    if len(results) and len(per): model_figures(results,per)
    sections = []
    sections.append('''<header><p class="eyebrow">Serova · AI × Science · 3 October 2026</p><h1>Can structural pretraining improve peptide–HLA stability prediction?</h1><p class="subtitle">A reproducible evaluation of 28,166 half-life measurements across 75 HLA class I alleles.</p></header>''')
    narrative = ROOT/'outputs/report_findings.html'
    if narrative.exists(): sections.append(narrative.read_text())
    else: sections.append('<section class="callout"><h2>Analysis in progress</h2><p>The audit is complete. Model results below are populated only from saved experiment outputs; missing experiments are not assigned scores.</p></section>')
    sections.append('''<h2>What the data actually contain</h2><p>The input has 28,166 unique allele–peptide pairs, 5,633 peptides and 75 alleles. All peptides are 9-mers. Exactly 5,679 targets (20.16%) equal zero; the median half-life is 1.1 h and the maximum is 256.7 h. The smallest positive value is 0.05 h. Zero prevalence is strongly associated with allele (Cramér’s V = 0.521).</p>
    <p>The original <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/">Rasmussen study</a> describes 28,166 experimental values before adding 1,000 artificial weak-binder zeros per allele. This file matches the experimental table’s size. A zero mass therefore does not establish synthetic padding. The assay’s exact zero convention and censoring threshold remain unknown. Here “nonzero” is an operational subset, not a claim that all other rows are unmeasured.</p>
    <p>Each source-paper value summarizes two experiments. No repeated allele–peptide pairs occur in this export, so raw replicate agreement and a numerical noise ceiling cannot be recovered. Cross-allele observations of one peptide are different biological complexes, not replicates. Every correlation figure states that the ceiling is unavailable.</p>''')
    sections.append(figure('audit_target_distribution.png','Target distribution. The marked floor is the chosen computational offset, not a validated assay detection limit.'))
    sections.append('''<p>Allele counts range from 7 to 1,070 (153-fold); seven alleles have fewer than 50 rows and twelve have fewer than 20 stable peptides. Between-allele differences account for 31.5% of the observed log-target variance descriptively. This motivates within-allele rather than pooled headline correlations.</p>''')
    sections.append(figure('audit_allele_distribution.png','Allele imbalance and heterogeneous zero prevalence.'))
    sections.append('''<h2>Correcting the structural hypothesis</h2><p>The supplied pseudosequence already includes six of the seven clamp positions named in the proposed rationale: 7, 59, 143, 147, 159 and 171; only 146 is omitted. Several of those six positions vary in this dataset. Thus the experiment tests transfer from structural pretraining and geometry; it cannot be presented as recovery of a clamp absent from the baseline.</p>
    <p>The joint subsequence problem has four solutions, not one: pseudosequence positions 18 and 24 have indistinguishable conserved-residue candidates. All four are saved. The published canonical 1-based mature-HLA indices reproduce every row:</p>
    <pre>7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171</pre>
    <p>Empirical high-stability motifs use half-life &gt;2 h and a 0.5 count per amino acid for smoothing. Unsmoothed anchor checks are consistent with expectation: A*02:01 has L/M at P2 in 76.6% and V/L at P9 in 77.9% of stable peptides; B*27:05 has R at P2 in 93.8%; A*03:01 has K/R at P9 in 88.6%. These descriptive motifs use the full table. Supervised PWMs are separately fitted within each training fold.</p>''')
    sections.append(figure('audit_empirical_motifs.png','High-stability empirical motifs; descriptive whole-table figures are not training features.'))
    sections.append(figure('audit_zero_anchor_composition.png','Zero versus nonzero anchor composition within allele. Zero rows do not uniformly resemble random nonbinders.'))
    sections.append('''<h2>Evaluation design</h2><p>Seed 20261003 fixes an allele-balanced 1,000-row pilot, 80/20 random and peptide-disjoint splits, and all 75 leave-one-allele-out folds. Random splitting reuses a training peptide in 5,219/5,634 test rows (92.6%); peptide-disjoint splitting has no peptide overlap. Random results are diagnostic only.</p>
    <p>The primary endpoint is the unweighted mean of within-allele Spearman correlations under leave-one-allele-out evaluation. Size-weighted within-allele means, pooled correlations, Pearson against log(t + 0.05), and AUC for t &gt;2 h are also reported on both all rows and the nonzero subset. SciPy averages tied ranks. The results table records eligible allele and row counts: correlations require at least three valid rows and variable targets; AUC requires both classes.</p>
    <p>A constant prediction has mathematically undefined correlation. To express its absence of ranking as requested, this implementation records an explicitly operational zero for constant predictions when the target varies. This convention applies to the allele-mean control; unavailable model predictions remain NA. It must not be interpreted as a computed Pearson/Spearman coefficient.</p>
    <p>LOAO tests unseen allele names, often close sequence neighbors: the median nearest pseudosequence differs at only 2/34 sites (maximum 9); B*14:01(C67S) and B*14:02(C67S) have identical pseudosequences. LOAO also permits peptides observed with other alleles. It is not a joint allele-and-peptide novelty test. The three splits have different test peptides and eligible allele populations; their score differences do not measure the causal effect of peptide reuse.</p>''')
    config = full/'config.json'
    if config.exists():
        sections.append('<details><summary>Saved supervised run configuration</summary><pre>'+html.escape(config.read_text())+'</pre></details>')
    sections.append('''<h2>Baseline ladder and target sensitivity</h2><p>Allele means and high-stability PWMs use training folds only. An allele mean falls back to the global training mean for an unseen allele; the categorical embedding and allele-specific PWM have no learned parameters for that allele, so their LOAO cells are explicitly unavailable. The pan-specific baseline concatenates a BLOSUM peptide encoding with a 34×20 one-hot pseudosequence.</p>
    <p>The allele-mean control illustrates the danger of pooling: on the random split, it obtains pooled Spearman 0.552 on all rows and 0.420 on nonzero rows without any peptide information. Its within-allele ranking is constant. Under LOAO, each global-mean fallback is fitted on a different training set; the pooled negative correlation of that control is a fold-composition artifact, not within-allele predictive ability. Ranking losses also leave allele-specific offsets unconstrained, making their pooled correlations particularly unsuitable as the headline.</p>
    <p>All learned comparisons use a shared 64/32 hidden-layer MLP, fixed optimizer and epoch budget. Input widths and therefore first-layer parameter counts differ, and each representation has a deterministic distinct initialization. Structural feature scaling is fitted on the training fold only. MSE uses log(t + 0.05). A one-sided squared censored surrogate is a sensitivity analysis with assumed c = 0.1 h and threshold log(c + 0.05): observed rows use squared error and zero rows incur only squared excess above that threshold. This is the requested hinge surrogate, not a Tobit likelihood. Within-allele pairwise ranking skips tied targets.</p>
    <p>The planned primary augmentation uses all 50 structural features with MSE; scalar, nine-position and 40-anchor features are separate ablations. After the ranking baseline outperformed MSE, an exploratory follow-up also trained the combined feature set with ranking loss in both scoring modes. Those results are compared with the ranking baseline. They were not used to replace or select the planned MSE result, and no held-out-label early stopping was used.</p>''')
    if len(results):
        head=results[(results.split=='leave_one_allele_out') & (results.metric=='spearman') & (results.aggregation.isin(['macro','size_weighted']))]
        head=head.pivot_table(index=['representation','loss'],columns=['subset','aggregation'],values='value',dropna=False).dropna(how='all').reset_index()
        head.columns=[' / '.join(str(v) for v in c if v) if isinstance(c,tuple) else c for c in head.columns]
        sections.append(table(head))
        sections.append(figure('model_loao.png','Primary endpoint, all and nonzero rows. Model/loss comparisons use the same fixed splits.'))
        sections.append(figure('model_splits.png','Diagnostic split comparison: reference controls and MSE heads. Missing LOAO PWM/embedding bars indicate unavailable predictions. Random-split results are not headline evidence.'))
        sections.append(figure('model_paired_alleles.png','Each point is one held-out allele. The diagonal indicates equal performance; it is not a noise ceiling.'))
        sections.append('<details><summary>Complete results, including unavailable cells and eligibility counts</summary>'+table(results)+'</details>')
    structural_text=ROOT/'outputs/structure/report_section.html'
    if structural_text.exists(): sections.append(structural_text.read_text())
    for path in sorted((ROOT/'outputs/structure').glob('structure*.png')):
        sections.append(figure(path,path.stem.replace('_',' ')))
    sections.append('''<h2>Interpretation limits</h2><p>Empirical motifs inherit the original affinity-selected peptide panel. Shared templates approximate geometry and do not repack allele-specific side chains; backbone conformations and anchor roles are not universal. LigandMPNN likelihood is sequence–structure compatibility, not a dissociation-rate model. A small conditional-versus-independent gap would concern these models and templates, not prove biological positional independence. Likewise, no incremental gain would not prove information equivalence.</p>
    <p>“Zero-shot” means no stability labels were used to fit the structural model. HLA structures or homologs may have appeared in foundation-model pretraining; structural pretraining independence has not been established. HLA scrambling is an out-of-distribution sensitivity check, not proof of a particular physical mechanism. The structural scorer also receives 182 HLA residues whereas the supervised baseline receives 34: without a full-HLA sequence-only control, gains cannot be attributed specifically to geometry rather than additional sequence context or pretrained priors. Fixed-budget single-seed model comparisons do not exhaust architectures or hyperparameters, and augmentation gains alone do not establish orthogonal biological information. A numerical replicate ceiling and a confirmed assay censoring limit are unavailable; the optional similarity-reduced peptide split was not run.</p>
    <h2>Reproducibility and files</h2><p>All results derive from local scripts and saved predictions. See <a href="results.csv">results.csv</a>, <a href="outputs/audit/audit.json">audit.json</a>, <a href="outputs/per_allele_metrics.csv">per-allele metrics</a>, <a href="notes/scientific_references.json">source notes</a>, and <a href="README.md">README</a>. Input SHA-256: <code>17c574c2c6d2746a238804bd73fa05279ae23e476be5df124dc084e116635658</code>. Split manifests, model configurations, logs, feature caches and dependency versions are saved in the project. Figures are embedded in this report for offline viewing and separately available as PNG/PDF artifacts.</p>
    <h2>Primary references</h2><ul><li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/">Rasmussen et al. (2016), pan-specific stability prediction and data provenance.</a></li><li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4341823/">Harndahl et al., dissociation assay.</a></li><li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC1949492/">NetMHCpan pseudosequence construction.</a></li><li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4498290/">HLA groove positions and substitutions.</a></li><li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC2211767/">HLA-B27-dependent peptide conformations.</a></li><li><a href="https://www.nature.com/articles/s41592-025-02626-1">LigandMPNN paper.</a> <a href="https://github.com/dauparas/LigandMPNN">Official code.</a></li></ul>''')
    style='''body{margin:auto;max-width:1120px;padding:52px 30px;font:17px/1.65 system-ui,sans-serif;color:#26333a;background:#fcfcf8}h1{font-size:43px;line-height:1.14;letter-spacing:-1.4px;max-width:1000px}h2{font-size:27px;margin-top:48px;color:#145961}a{color:#126d78}.eyebrow{font-size:13px;text-transform:uppercase;letter-spacing:2px;color:#65767b}.subtitle{font-size:21px;color:#5c6e74}.callout{background:#edf5f4;border-left:5px solid #177e89;padding:8px 24px;margin:32px 0}figure{margin:32px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#647379}table{font-size:12px;display:block;overflow-x:auto;border-collapse:collapse;line-height:1.45}th,td{padding:7px 9px;text-align:right;border-bottom:1px solid #d9e2e2}th{background:#eaf1f0}pre{white-space:pre-wrap;background:#eef2f1;padding:18px;font-size:12px}details{margin:22px 0}summary{cursor:pointer;font-weight:600}@media print{body{padding:0;font-size:11px}h1{font-size:27px}figure,table{break-inside:avoid}details{display:block}}'''
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Peptide–HLA stability: empirical evaluation</title><style>'+style+'</style><body>'+''.join(sections)+'</body></html>'
    (ROOT/'report.html').write_text(doc)
    print('Wrote report.html')

if __name__=='__main__': main()
