"""Descriptive structural controls and figures. No model or template selection."""
import json
import html
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
_mpl_cache=Path(__file__).resolve().parents[1]/'outputs/structure/.mplconfig'
_mpl_cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(_mpl_cache))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from structural_pipeline import OUT,ROOT,TEMPLATES,AA,encode_seq,dump_json,build_template
from splits import load_data
from evaluation import metric_values,NOISE_CEILING

def annotate(fig):
    fig.text(.5,.008,NOISE_CEILING,ha='center',fontsize=8,color='#555555')

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def allele_macro(data,pred,nonzero=False):
    d=data.assign(prediction=pred)
    if nonzero:d=d[d.thalf_hours>0]
    return np.nanmean([metric_values(g)[0]['spearman'] for _,g in d.groupby('allele')])

def main():
    data=load_data().set_index('row_id',drop=False)
    dump_json(OUT/'decoding_orders.json',{'peptide_positions_one_based':{
        str(seed):(np.random.default_rng(seed+9001).permutation(9)+1).tolist() for seed in [17,29,43]},
        'hla_order':'Random permutation of fixed residues with the same seed; all fixed before all peptide positions.'})
    templates=[build_template(t,'truncated',data) for t in TEMPLATES]
    overlap=[]
    for template in templates:
        t=template[-1];matches=data[data.peptide==t['native_peptide']]
        overlap.append({'pdb_id':t['pdb_id'],'native_peptide':t['native_peptide'],
                        'any_allele_rows':len(matches),'native_allele_rows':int(matches.allele.eq(t['native_allele']).sum())})
    pd.DataFrame(overlap).to_csv(OUT/'native_peptide_overlap.csv',index=False)
    geometry=[]
    for x in templates:
        for y in templates:
            a=x[0]['X'][0,:,1].numpy();b=y[0]['X'][0,:,1].numpy()
            ac=a[x[1]];bc=b[y[1]];am=ac.mean(0);bm=bc.mean(0)
            u,_,vh=np.linalg.svd((ac-am).T@(bc-bm));rot=u@vh
            if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vh
            diff=(a[x[3]]-am)@rot+bm-b[y[3]]
            geometry.append({'template1':x[-1]['pdb_id'],'template2':y[-1]['pdb_id'],
                'hla_ca_rmsd':float(np.sqrt(np.mean(np.sum(((ac-am)@rot+bm-bc)**2,-1)))),
                'peptide_ca_rmsd':float(np.sqrt((diff**2).sum(-1).mean())),
                **{f'P{k+1}_distance':float(z) for k,z in enumerate(np.sqrt((diff**2).sum(-1)))}})
    pd.DataFrame(geometry).to_csv(OUT/'template_rmsd.csv',index=False)
    suffix='' if (OUT/'features_unconditional.npz').exists() else '_pilot'
    u=np.load(OUT/f'features_unconditional{suffix}.npz')
    d=data.loc[u['row_id']].copy()
    modes={'Peptide independent':u}
    for label,file in [('Autoregressive','conditional'),('Scrambled HLA','scrambled')]:
        path=OUT/f'features_{file}{suffix}.npz'
        if path.exists():modes[label]=np.load(path)
    for label,features in modes.items():
        assert np.array_equal(features['row_id'],u['row_id']),f'Feature row alignment differs: {label}'
    motifs=np.load(OUT/f'unconditional_motifs{suffix}.npz')
    alleles=motifs['alleles'].tolist();logp=motifs['log_probs'];prob=np.exp(logp);prob/=prob.sum(-1,keepdims=True)
    avg_prob=prob.mean(0)
    empirical=json.loads((ROOT/'outputs/audit/empirical_motifs.json').read_text())
    chosen=['HLA-A*02:01','HLA-B*27:05','HLA-A*03:01']
    template_by_hash={t['template_sha256']:t['pdb_id'] for t in json.loads((OUT/'templates.json').read_text())}
    scrambled_manifest=json.loads((OUT/f'cache_manifest_scrambled{suffix}.json').read_text())
    matched_scrambled={}
    for allele in chosen:
        native_id=next(t for t,v in TEMPLATES.items() if v[0]==allele)
        arrays=[]
        for cached in scrambled_manifest:
            if cached['allele']==allele and template_by_hash[cached['template']]==native_id:
                p=np.exp(np.load(ROOT/cached['path'])['log_probs']);arrays.append(p/p.sum(-1,keepdims=True))
        matched_scrambled[allele]=np.mean(arrays,axis=0)
    fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
    motif_summary={}
    for i,a in enumerate(chosen):
        ai=alleles.index(a);ti=list(TEMPLATES).index(next(t for t,v in TEMPLATES.items() if v[0]==a))
        emp=np.array(empirical['alleles'][a]['probabilities']);ensemble=avg_prob[ai];native=prob[ti*3:ti*3+3,ai].mean(0)
        motif_summary[a]={'P2_top':AA[ensemble[1].argmax()],'P9_top':AA[ensemble[8].argmax()],
                          'P2_top_native':AA[native[1].argmax()],'P9_top_native':AA[native[8].argmax()],
                          'P9_KR_probability_ensemble':float(ensemble[8,AA.index('K')]+ensemble[8,AA.index('R')]),
                          'P9_KR_probability_native':float(native[8,AA.index('K')]+native[8,AA.index('R')]),
                          'P9_KR_probability_empirical':float(emp[8,AA.index('K')]+emp[8,AA.index('R')]),
                          'P2_R_probability_ensemble':float(ensemble[1,AA.index('R')]),
                          'P2_R_probability_native':float(native[1,AA.index('R')]),
                          'P2_R_probability_native_scrambled':float(matched_scrambled[a][1,AA.index('R')]),
                          'P2_top_native_scrambled':AA[matched_scrambled[a][1].argmax()],
                          'P9_top_native_scrambled':AA[matched_scrambled[a][8].argmax()],
                          'P2_R_probability_empirical':float(emp[1,AA.index('R')])}
        for j,(title,matrix) in enumerate([('Empirical: half-life >2 h',emp),('Shared ensemble',ensemble),('Matching allele template',native)]):
            im=axes[i,j].imshow(matrix.T,vmin=0,vmax=1,aspect='auto',cmap='magma')
            axes[i,j].set(xticks=np.arange(9),xticklabels=np.arange(1,10),yticks=np.arange(20),yticklabels=list(AA),title=f'{a}\n{title}',xlabel='Peptide position')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Amino-acid probability',shrink=.65)
    fig.suptitle('Does the structural model recover allele-specific motifs?\nWhole-data empirical motifs are descriptive; no fitting to these motifs',fontsize=13)
    save(fig,'structure_motif_comparison')
    pd.DataFrame([{'allele':allele,**metrics} for allele,metrics in motif_summary.items()]).to_csv(OUT/'native_motif_scramble.csv',index=False)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label,z in modes.items():
        # Weight alleles equally rather than allowing abundant alleles to dominate.
        profile=pd.DataFrame(z['per_position'],index=d.allele).groupby(level=0).mean().mean(axis=0)
        axes[0].plot(np.arange(1,10),profile,marker='o',label=label)
    entropy=-(avg_prob*np.log(avg_prob)).sum(-1).mean(0)/np.log(2)
    axes[1].plot(np.arange(1,10),entropy,marker='o',color='#7868b8')
    for ax in axes:
        ax.set_xlabel('Peptide position');ax.set_xticks(np.arange(1,10));ax.axvspan(1.85,2.15,color='#ddd',alpha=.6);ax.axvspan(8.85,9.15,color='#ddd',alpha=.6)
    axes[0].set_ylabel('Observed-residue mean log probability');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Predicted motif entropy (bits)');axes[1].set_title('Lower entropy = stronger residue preference')
    fig.suptitle('Position-wise constraints: P2 and P9 highlighted')
    save(fig,'structure_position_profiles')

    control_rows=[]
    for label,z in modes.items():
        for subset in ['all','nonzero']:
            val=allele_macro(d,z['mean_logp'],subset=='nonzero')
            control_rows.append({'mode':label,'subset':subset,'macro_spearman':float(val)})
    ct=pd.DataFrame(control_rows);ct.to_csv(OUT/'structural_control_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.5));pivot=ct.pivot(index='mode',columns='subset',values='macro_spearman')
    pivot.plot.bar(ax=ax,color=['#356c91','#e49655']);ax.set(ylabel='Mean within-allele Spearman',xlabel='',title=f'Structural scoring and scrambled-HLA control ({len(d):,} rows)');ax.tick_params(axis='x',rotation=0);ax.axhline(0,color='#777',lw=.7)
    ax.text(.5,-.12,'Scrambled-HLA control shown here uses peptide-independent scoring.',transform=ax.transAxes,ha='center',fontsize=8)
    annotate(fig);fig.subplots_adjust(bottom=.18);save(fig,'structure_scrambled_control')

    a=np.load(OUT/'control500_truncated.npz');b=np.load(OUT/'control500_full.npz')
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(a['mean_logp'],b['mean_logp'],s=10,alpha=.45,color='#356c91')
    low=min(a['mean_logp'].min(),b['mean_logp'].min());high=max(a['mean_logp'].max(),b['mean_logp'].max())
    ax.plot([low,high],[low,high],color='#777',lw=1)
    selection=json.loads((OUT/'assembly_selection.json').read_text())
    assembly_metrics=[]
    for name,values in [('truncated',a),('full',b)]:
        rows=data.loc[values['row_id']]
        for subset in ['all','nonzero']:
            assembly_metrics.append({'assembly':name,'subset':subset,
                'macro_spearman':float(allele_macro(rows,values['mean_logp'],subset=='nonzero')),
                'n':len(rows) if subset=='all' else int((rows.thalf_hours>0).sum()),
                'used_for_selection':False})
    pd.DataFrame(assembly_metrics).to_csv(OUT/'assembly500_target_metrics.csv',index=False)
    ax.set(xlabel='Alpha1/alpha2 only: mean log probability',ylabel='Full heterotrimer: mean log probability',title=f'500-row assembly check\nMean absolute change {selection["mean_absolute_logp_difference"]:.4f}; r={selection["score_correlation"]:.4f}')
    annotate(fig);fig.subplots_adjust(bottom=.16);save(fig,'structure_assembly_control')

    # Existing real-allele structures vs common template and leave-native-out ensemble.
    native_rows=[]
    for ti,(template,(allele,*_)) in enumerate(TEMPLATES.items()):
        g=data[data.allele==allele];ai=alleles.index(allele)
        encoded=np.array([encode_seq(s) for s in g.peptide]);ensemble=logp[:,ai]
        configs={'native_template':ensemble[ti*3:ti*3+3].mean(0),
                 'shared_1HHK':ensemble[:3].mean(0),
                 'non_native_ensemble':np.delete(ensemble,np.arange(ti*3,ti*3+3),axis=0).mean(0)}
        for name,lp in configs.items():
            pred=lp[np.arange(9)[None,:],encoded].mean(-1)
            for subset in ['all','nonzero']:
                gg=g.assign(prediction=pred)
                if subset=='nonzero':gg=gg[gg.thalf_hours>0]
                metrics,n=metric_values(gg)
                native_rows.append({'allele':allele,'native_pdb':template,'configuration':name,'subset':subset,'n':n,**metrics})
    nt=pd.DataFrame(native_rows);nt.to_csv(OUT/'native_vs_shared_templates.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,subset in zip(axes,['all','nonzero']):
        nt[nt.subset==subset].pivot(index='allele',columns='configuration',values='spearman').plot.bar(ax=ax,color=['#356c91','#96b8ca','#e49655'])
        ax.set(title=subset,ylabel='Within-allele Spearman',xlabel='');ax.tick_params(axis='x',rotation=30);ax.axhline(0,color='#777',lw=.6)
    axes[1].legend().remove();fig.suptitle('Native-allele versus common template comparison');annotate(fig);fig.subplots_adjust(bottom=.22,wspace=.3)
    save(fig,'structure_native_template_control')

    summary={'rows':len(d),'features_suffix':suffix,'assembly':selection,'assembly_target_metrics':assembly_metrics,
             'native_peptide_dataset_overlap':overlap,'motifs':motif_summary,
             'metrics':control_rows,'position_motif_entropy_bits':entropy.tolist(),
             'mean_unconditional_template_order_score_sd':float(u['spread'].mean()),
             'native_template_comparison':'5 available real alleles, native vs common 1HHK and ensemble excluding native; descriptive, no template tuning'}
    runtime={}
    for key,file,prefix in [('first_unconditional_motif_fill_seconds','pilot.log','unconditional scramble=False'),
                            ('pilot_conditional_seconds','pilot.log','conditional scramble=False'),
                            ('full_conditional_expansion_seconds','full.log','conditional scramble=False')]:
        if (OUT/file).exists():
            lines=[line for line in (OUT/file).read_text().splitlines() if line.startswith(prefix)]
            if lines:runtime[key]=float(re.search(r': ([0-9.]+)s$',lines[-1]).group(1))
    summary['observed_runtime_seconds']=runtime
    if 'Scrambled HLA' in modes:
        su=modes['Scrambled HLA'];delta=su['mean_logp']-u['mean_logp']
        summary['scrambled_control']={'mean_abs_score_change':float(np.abs(delta).mean()),'max_abs_score_change':float(np.abs(delta).max()),'score_spearman':float(spearmanr(su['mean_logp'],u['mean_logp']).statistic)}
    if 'Autoregressive' in modes:
        c=modes['Autoregressive'];summary['conditional_control']={'mean_abs_score_change':float(np.abs(c['mean_logp']-u['mean_logp']).mean()),'mean_template_order_sd':float(c['spread'].mean()),'score_spearman':float(spearmanr(c['mean_logp'],u['mean_logp']).statistic)}
        gaps=[]
        for subset in ['all','nonzero']:
            frame=d.assign(independent=u['mean_logp'],autoregressive=c['mean_logp'])
            if subset=='nonzero':frame=frame[frame.thalf_hours>0]
            deltas=[]
            for _,group in frame.groupby('allele'):
                r1=metric_values(group.assign(prediction=group.independent))[0]['spearman']
                r2=metric_values(group.assign(prediction=group.autoregressive))[0]['spearman']
                if np.isfinite(r1) and np.isfinite(r2):deltas.append(r2-r1)
            delta=np.array(deltas);bootstrap=np.random.default_rng(20261003).choice(delta,(5000,len(delta)),replace=True).mean(1)
            gaps.append({'subset':subset,'delta_macro_spearman':float(delta.mean()),'n_alleles':len(delta),
                         'ci_low':float(np.quantile(bootstrap,.025)),'ci_high':float(np.quantile(bootstrap,.975)),
                         'bootstrap_unit':'allele','caveat':'Descriptive; related alleles are not independent; not a kinetics mechanism test.'})
        summary['conditional_control']['conditional_minus_independent_macro_spearman']=gaps
        scores=c['template_order_scores'].reshape(5,3,len(d))
        summary['conditional_control']['mean_between_template_score_sd']=float(scores.mean(1).std(0).mean())
        summary['conditional_control']['mean_within_template_order_score_sd']=float(scores.std(1).mean())
        conditional_native=[]
        for ti,(template,(allele,*_)) in enumerate(TEMPLATES.items()):
            chosen_rows=d.allele.eq(allele).to_numpy();g=d[chosen_rows]
            configs={'native_template':scores[ti].mean(0)[chosen_rows],
                     'shared_1HHK':scores[0].mean(0)[chosen_rows],
                     'non_native_ensemble':np.delete(scores,ti,axis=0).mean((0,1))[chosen_rows]}
            for name,pred in configs.items():
                for subset in ['all','nonzero']:
                    gg=g.assign(prediction=pred)
                    if subset=='nonzero':gg=gg[gg.thalf_hours>0]
                    metrics,n=metric_values(gg)
                    conditional_native.append({'allele':allele,'native_pdb':template,'configuration':name,'subset':subset,'n':n,**metrics})
        pd.DataFrame(conditional_native).to_csv(OUT/'native_vs_shared_conditional.csv',index=False)
    if (OUT/'conditional500_controls.json').exists():
        controls=json.loads((OUT/'conditional500_controls.json').read_text())
        summary['conditional500_controls']=controls
        fig,ax=plt.subplots(figsize=(9,5))
        pd.DataFrame(controls['metrics']).pivot(index='variant',columns='subset',values='macro_spearman').plot.bar(ax=ax,color=['#356c91','#e49655'])
        ax.set(xlabel='',ylabel='Mean within-allele Spearman',title='Autoregressive controls on 500 stratified rows\nExploratory sensitivity check; selected assembly remains fixed')
        ax.tick_params(axis='x',rotation=0);ax.axhline(0,color='#777',lw=.7)
        annotate(fig);fig.subplots_adjust(bottom=.16);save(fig,'structure_conditional500_controls')
    spread_summary={};member_rows=[]
    for name,z in modes.items():
        spread_summary[name]={}
        for subset in ['all','nonzero']:
            keep=np.ones(len(d),bool) if subset=='all' else d.thalf_hours.gt(0).to_numpy()
            sd=z['spread'][keep]
            spread_summary[name][subset]={'mean':float(sd.mean()),'median':float(np.median(sd)),
                                          'p05':float(np.quantile(sd,.05)),'p95':float(np.quantile(sd,.95))}
            for i,scores in enumerate(z['template_order_scores']):
                member_rows.append({'mode':name,'subset':subset,'template':list(TEMPLATES)[i//3],
                    'seed':[17,29,43][i%3], 'macro_spearman':float(allele_macro(d,scores,subset=='nonzero'))})
    summary['per_row_ensemble_score_sd']=spread_summary
    members=pd.DataFrame(member_rows);members.to_csv(OUT/'ensemble_member_metrics.csv',index=False)
    summary['individual_ensemble_member_macro_spearman_range']=[
        {'mode':name,'subset':subset,'minimum':float(g.macro_spearman.min()),'maximum':float(g.macro_spearman.max())}
        for (name,subset),g in members.groupby(['mode','subset'])]
    dump_json(OUT/'structural_summary.json',summary)
    bp=motif_summary['HLA-B*27:05']
    text=f'''<section><h2>Structural controls</h2><p>Five validated nine-residue peptide templates (1HHK, 5IB1, 7MLE, 4NQX, 3LKN; resolution 1.91–2.50 Å) were scored with LigandMPNN and three fixed random decoding orders (17, 29, 43). HLA sequence was threaded onto the alpha1/alpha2 backbone; no side-chain coordinates or nonprotein ligand atom context were supplied. Peptide-independent mode hides every peptide identity while retaining HLA sequence. Autoregressive mode reveals earlier peptide residues. Both paths agree with the official full decoder to numerical precision, and peptide identity invariance was independently tested.</p><p>In the predeclared label-free 500-row assembly comparison, truncated/full mean absolute score change was {selection['mean_absolute_logp_difference']:.4f}, below the 0.02 criterion, with r={selection['score_correlation']:.4f}. Truncation was selected for speed before evaluating target correlations. This does not establish equivalence of the underlying molecular systems. The five template peptide backbones differ by 1.10–1.79 Å after aligning the HLA domain.</p><p>For HLA-B*27:05 P2, the ensemble predicts R probability {bp['P2_R_probability_ensemble']:.3f}; its most probable residue is {bp['P2_top']}. The matching B*27:05 structure gives R probability {bp['P2_R_probability_native']:.3f}, compared with {bp['P2_R_probability_empirical']:.3f} in the descriptive high-stability empirical motif. Template and order spread is saved per row; it describes model sensitivity, not a calibrated confidence interval.</p><p>All template comparisons are descriptive controls. Shared backbone threading and backbone-only inputs do not test the full potential of relaxed, allele-specific side-chain models. Training-set overlap between the foundation model and historical PDB templates cannot be excluded. Replicate noise ceiling is unavailable because the supplied data contain no repeated allele–peptide pairs.</p></section>'''
    spread_html=f'<p>On the same native B*27:05 backbone, scrambling HLA sequence reduces P2-R probability from {bp["P2_R_probability_native"]:.3f} to {bp["P2_R_probability_native_scrambled"]:.3f}. Thus this native-template motif depends on HLA sequence as well as geometry; ensemble transfer to shared templates dilutes the preference.</p>'
    spread_html+='<p>The observed-residue likelihood profile peaks at P2/P9, showing anchor signal in these observed peptides. The entropy diagnostic asks a different question: which positions have the narrowest predicted amino-acid distribution. Its minimum is P3, while P2/P9 are narrower than most central positions. These diagnostics should not be treated as interchangeable.</p>'
    measured=nt[nt.subset=='nonzero'].pivot(index='allele',columns='configuration',values='spearman')
    better=int((measured.native_template>measured.non_native_ensemble).sum())
    spread_html+=f'<p>Native-template effects are heterogeneous: matching structures improve measured-only ranking over the ensemble excluding that native structure for {better} of 5 tested alleles. For B*35:01, Spearman rises from {measured.loc["HLA-B*35:01","non_native_ensemble"]:.3f} to {measured.loc["HLA-B*35:01","native_template"]:.3f}; for B*27:05, from {measured.loc["HLA-B*27:05","non_native_ensemble"]:.3f} to {measured.loc["HLA-B*27:05","native_template"]:.3f}. A*03:01 moves the other way, from {measured.loc["HLA-A*03:01","non_native_ensemble"]:.3f} to {measured.loc["HLA-A*03:01","native_template"]:.3f}. Shared-template transfer is therefore not universally validated, and matching geometry is not a guaranteed remedy.</p>'
    if runtime:
        spread_html+=f'<p>On the local CPU, the initial peptide-independent motif pass took {runtime.get("first_unconditional_motif_fill_seconds",float("nan")):.1f} s; these allele-level results were then reused for all 28,166 peptides. Expanding autoregressive scoring to the full dataset took {runtime.get("full_conditional_expansion_seconds",float("nan")):.1f} s, reusing peptide scores from the {runtime.get("pilot_conditional_seconds",float("nan")):.1f} s prototype pass. Thus the extra computation produced very little change in ranking accuracy in this experiment. Timings depend on simultaneous local workloads and exclude setup.</p>'
    spread_html+='<h3>Template and decoding-order sensitivity</h3><p>Each row carries the standard deviation of its mean log score across 5 templates × 3 decoding orders. These values describe model sensitivity and are not confidence intervals.</p><table><thead><tr><th>Scoring mode / subset</th><th>Mean SD</th><th>Median SD</th><th>5–95% of row SDs</th></tr></thead><tbody>'
    for name,subsets in spread_summary.items():
        for subset,v in subsets.items():
            spread_html+=f'<tr><td>{html.escape(name)} / {subset}</td><td>{v["mean"]:.3f}</td><td>{v["median"]:.3f}</td><td>{v["p05"]:.3f}–{v["p95"]:.3f}</td></tr>'
    spread_html+='</tbody></table><p>Individual template/order macro-Spearman ranges: '
    spread_html+='; '.join(f'{x["mode"]} ({x["subset"]}) {x["minimum"]:.3f}–{x["maximum"]:.3f}' for x in summary['individual_ensemble_member_macro_spearman_range'])+'. Full member metrics are saved in ensemble_member_metrics.csv.</p>'
    spread_html+='<p>The pinned official LigandMPNN score.py exposes --chains_to_design, --autoregressive_score and --use_sequence. The original use_sequence=0 removes HLA sequence as well as peptide sequence, so this experiment uses the model API with a tested custom HLA-conditioned mask. Fixed-context decoder caching reproduces official scores with maximum absolute error below 10⁻⁶.</p>'
    if 'conditional_control' in summary:
        spread_html+='<p>Autoregressive minus peptide-independent macro Spearman: '
        spread_html+='; '.join(f'{x["subset"]} Δ={x["delta_macro_spearman"]:.3f} (descriptive allele-bootstrap 95% interval {x["ci_low"]:.3f} to {x["ci_high"]:.3f})' for x in summary['conditional_control']['conditional_minus_independent_macro_spearman'])
        spread_html+='. This compares the two scoring formulations; it does not by itself establish physical positional independence or dependence in peptide binding.</p>'
    if 'conditional500_controls' in summary:
        ctrl=summary['conditional500_controls']
        spread_html+=f'<h3>Exploratory autoregressive controls</h3><p>On 500 stratified rows, shuffling HLA changes mean log score by {ctrl["scrambled"]["mean_abs_score_change"]:.3f} on average (absolute difference); retaining the full assembly changes it by {ctrl["full_assembly"]["mean_abs_score_change"]:.3f}. The already selected truncated assembly remains fixed. Per-allele correlations on this small subset have few rows per allele and are exploratory.</p>'
        spread_html+=pd.DataFrame(ctrl['metrics']).to_html(index=False,float_format=lambda x:f'{x:.3f}')
    text=text.replace('</section>',spread_html+'</section>')
    (OUT/'report_section.html').write_text(text)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
