"""Build numeric report findings from completed full-data experiment artifacts."""
from pathlib import Path
import json
import pandas as pd

def main():
    r=pd.read_csv('results.csv')
    p=pd.read_csv('outputs/paired_comparisons.csv')
    def get(rep,loss,subset,metric='spearman',agg='macro'):
        q=r[(r.representation==rep)&(r.loss==loss)&(r.subset==subset)&(r.metric==metric)&(r.aggregation==agg)&(r.split=='leave_one_allele_out')]
        if len(q)!=1 or pd.isna(q.value.iloc[0]): raise ValueError(f'Missing completed result: {rep}, {loss}, {subset}, {metric}')
        return q.value.iloc[0]
    rows=[]
    models=[('Structural score · independent','ligandmpnn_unconditional','zero_shot'),
            ('Structural score · autoregressive','ligandmpnn_conditional','zero_shot'),
            ('BLOSUM + pseudosequence','blosum_pseudoseq','mse'),
            ('Baseline + independent features','baseline_plus_unconditional_all','mse'),
            ('Baseline + autoregressive features','baseline_plus_conditional_all','mse'),
            ('BLOSUM + pseudosequence','blosum_pseudoseq','ranking'),
            ('Baseline + independent features','baseline_plus_unconditional_all','ranking'),
            ('Baseline + autoregressive features','baseline_plus_conditional_all','ranking')]
    for label,rep,loss in models:
        rows.append({'Model':label,'Loss':loss,'Spearman · all':get(rep,loss,'all'),
                     'Spearman · nonzero':get(rep,loss,'nonzero'),
                     'Weighted · all':get(rep,loss,'all',agg='size_weighted'),
                     'Weighted · nonzero':get(rep,loss,'nonzero',agg='size_weighted')})
    head=pd.DataFrame(rows)
    head.to_csv('outputs/headline_results.csv',index=False)
    fmt=lambda x:f'{x:.3f}'
    sections=['<section class="callout"><h2>Observed result</h2>']
    conclusion=Path('outputs/conclusion.html')
    if conclusion.exists():sections.append(conclusion.read_text())
    else:sections.append('<p>All values below are completed leave-one-allele-out results on the full dataset. They are mean correlations within held-out alleles, rather than a pooled correlation across alleles.</p>')
    sections.append(head.to_html(index=False,border=0,float_format=fmt))
    sections.append('<p>All 75 alleles are eligible in both subsets for these primary comparisons. “All” contains 28,166 rows; “nonzero” contains 22,487. The combined feature vector contains one scalar, nine observed-residue log probabilities and 40 anchor probabilities. MSE is the planned primary comparison; ranking augmentation is an exploratory follow-up. No numerical replicate noise ceiling can be estimated from this export.</p></section>')
    sections.append('<h2>Paired changes across held-out alleles</h2>')
    keep=p.variant.str.endswith('_all | mse')|p.variant.str.endswith('_all | ranking')|p.variant.eq('blosum_pseudoseq | ranking')|p.variant.eq('blosum_pseudoseq | censored_hinge')|p.variant.eq('ligandmpnn_conditional | zero_shot')
    q=p.loc[keep,['subset','baseline','variant','delta_macro_spearman','ci_low','ci_high','alleles_improved','n_alleles']].copy()
    sections.append(q.to_html(index=False,border=0,float_format=fmt))
    sections.append('<p>Intervals are descriptive 95% percentile intervals from 5,000 paired bootstrap resamples of alleles. They do not include retraining-seed uncertainty, account for allele relatedness or overlapping training folds, or correct for multiple comparisons. They should not be read as confirmatory significance tests. No ablation was selected to replace the combined-feature primary result.</p>')
    secondary=[]
    for label,rep,loss in models:
        secondary.append({'Model':label,'Loss':loss,
                          'Pearson log · all':get(rep,loss,'all',metric='pearson_log'),
                          'Pearson log · nonzero':get(rep,loss,'nonzero',metric='pearson_log'),
                          'AUC >2h · all':get(rep,loss,'all',metric='auc_gt_2h'),
                          'AUC >2h · nonzero':get(rep,loss,'nonzero',metric='auc_gt_2h')})
    sections.append('<h2>Secondary endpoints</h2>'+pd.DataFrame(secondary).to_html(index=False,border=0,float_format=fmt))
    sections.append('<p>These are also unweighted within-allele means. AUC excludes alleles without both threshold classes in the evaluated subset; exact eligibility counts are in results.csv. Structural likelihoods and ranking outputs are uncalibrated scores, not predicted hours.</p>')
    sections.append('<h2>What changed with the loss</h2><p>Pairwise ranking changes the label-only baseline from '
                    f'{get("blosum_pseudoseq","mse","all"):.3f}/{get("blosum_pseudoseq","mse","nonzero"):.3f} to '
                    f'{get("blosum_pseudoseq","ranking","all"):.3f}/{get("blosum_pseudoseq","ranking","nonzero"):.3f} '
                    '(all/nonzero macro Spearman). The one-sided censoring surrogate gives '
                    f'{get("blosum_pseudoseq","censored_hinge","all"):.3f}/{get("blosum_pseudoseq","censored_hinge","nonzero"):.3f}. '
                    'These objective comparisons use the same representation and fixed training budget. The assumed zero threshold remains unverified.</p>')
    sections.append('<p>Random test rows reuse training peptides in 92.6% of cases. Nonetheless, this particular random split does not give higher mean per-allele performance than the peptide-disjoint split. The result demonstrates overlap, not a measured causal inflation in accuracy; the test compositions and eligible allele counts differ.</p>')
    Path('outputs/report_findings.html').write_text(''.join(sections))
    print('Wrote report findings and headline_results.csv')

if __name__=='__main__': main()
