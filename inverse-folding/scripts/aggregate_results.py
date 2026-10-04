"""Assemble actually completed full-data runs and zero-shot scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluation import evaluate_predictions
from splits import load_data, make_splits, SEED

def main():
    data=load_data()
    pieces=[]
    included=[]
    for directory in ['full','full_unconditional','full_conditional','full_unconditional_ranking','full_conditional_ranking']:
        path=Path('outputs/baselines')/directory/'predictions.csv.gz'
        if not path.exists(): continue
        frame=pd.read_csv(path)
        # Partial run checkpoints do not become headline metrics.
        for key,g in frame.groupby(['representation','loss']):
            lo=g[g.split=='leave_one_allele_out']
            if len(lo)!=len(data) or lo.allele.nunique()!=75:
                print('Skipping incomplete representation',directory,key,len(lo));continue
            pieces.append(g)
            included.append({'path':str(path),'representation':key[0],'loss':key[1]})
    for mode in ['unconditional','conditional','scrambled']:
        path=Path(f'outputs/structure/features_{mode}.npz')
        if not path.exists(): continue
        ar=np.load(path)
        if len(ar['row_id'])!=len(data):
            print('Skipping partial structural archive',path);continue
        assert len(np.unique(ar['row_id']))==len(data)
        lookup=dict(zip(ar['row_id'].astype(int),ar['mean_logp'].reshape(-1)))
        for split,fold,_,ids in make_splits(data):
            f=data.set_index('row_id',drop=False).loc[ids,['row_id','allele','thalf_hours']].copy()
            f['prediction']=[lookup[i] for i in ids]
            f['representation']=f'ligandmpnn_{mode}'
            f['loss']='zero_shot';f['split']=split;f['fold']=fold
            pieces.append(f)
        included.append({'path':str(path),'representation':f'ligandmpnn_{mode}','loss':'zero_shot'})
    if not pieces: raise SystemExit('No complete full-data run available.')
    pred=pd.concat(pieces,ignore_index=True)
    assert not pred.duplicated(['representation','loss','split','row_id']).any()
    results,per=evaluate_predictions(pred,'outputs')
    results.to_csv('results.csv',index=False)
    (Path('outputs')/'included_runs.json').write_text(json.dumps(included,indent=2))
    # Paired bootstrap of held-out allele scores, with training uncertainty caveat.
    comparisons=[]
    for subset in ['all','nonzero']:
        g=per[(per.split=='leave_one_allele_out') & (per.subset==subset) & (per.metric=='spearman')].copy()
        g['model']=g.representation+' | '+g.loss
        p=g.pivot(index='allele',columns='model',values='value')
        pairs=[]
        baseline='blosum_pseudoseq | mse'
        if baseline in p:
            pairs += [(baseline,v) for v in p.columns if (v.startswith('baseline_plus_') and v.endswith(' | mse')) or v.startswith('blosum_pseudoseq |') if v!=baseline]
        rank_baseline='blosum_pseudoseq | ranking'
        if rank_baseline in p:
            pairs += [(rank_baseline,v) for v in p.columns if v.startswith('baseline_plus_') and v.endswith(' | ranking')]
        if all(v in p for v in ['ligandmpnn_unconditional | zero_shot','ligandmpnn_conditional | zero_shot']):
            pairs.append(('ligandmpnn_unconditional | zero_shot','ligandmpnn_conditional | zero_shot'))
        for loss in ['mse','ranking']:
            independent=f'baseline_plus_unconditional_all | {loss}'
            autoregressive=f'baseline_plus_conditional_all | {loss}'
            if independent in p and autoregressive in p:
                pairs.append((independent,autoregressive))
        for b,v in pairs:
            q=p[[b,v]].dropna()
            d=(q[v]-q[b]).to_numpy()
            rng=np.random.default_rng(SEED)
            ci=np.quantile(rng.choice(d,(5000,len(d)),replace=True).mean(1),[.025,.975])
            comparisons.append({'subset':subset,'baseline':b,'variant':v,'n_alleles':len(d),
                                'delta_macro_spearman':d.mean(),'ci_low':ci[0],'ci_high':ci[1],
                                'alleles_improved':int((d>0).sum()),'bootstrap_draws':5000,'seed':SEED})
    pd.DataFrame(comparisons).to_csv('outputs/paired_comparisons.csv',index=False)
    Path('outputs/paired_comparisons_note.txt').write_text('Descriptive percentile bootstrap over alleles. Related alleles and overlapping training folds are not independent; no training-seed uncertainty or multiplicity correction is included. These intervals are not confirmatory significance tests.\n')
    print(f'Wrote {len(results)} metric cells from {len(included)} runs; {len(pred)} predictions.')

if __name__=='__main__':main()
