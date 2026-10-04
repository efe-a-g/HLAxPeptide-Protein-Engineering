"""Check completed deliverables and bind them to a compact hash manifest."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    data=pd.read_csv('resources/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv')
    r=pd.read_csv('results.csv')
    expected={('allele_mean','reference'),('pwm','reference')}
    expected|={(rep,loss) for rep in ['allele_embedding','blosum_pseudoseq'] for loss in ['mse','censored_hinge','ranking']}
    expected|={(f'baseline_plus_{mode}_{feature}','mse') for mode in ['unconditional','conditional'] for feature in ['mean_logp','logp_per_position','anchor_softmax','all']}
    expected|={(f'baseline_plus_{mode}_all','ranking') for mode in ['unconditional','conditional']}
    expected|={(f'ligandmpnn_{mode}','zero_shot') for mode in ['unconditional','conditional','scrambled']}
    assert set(map(tuple,r[['representation','loss']].drop_duplicates().to_numpy()))==expected
    assert len(r)==len(expected)*3*2*3*3
    keys=['representation','loss','split','subset','aggregation','metric']
    assert not r.duplicated(keys).any()
    assert (r.status.eq('ok')==r.value.notna()).all()
    assert r.value.dropna().between(-1.000001,1.000001).all()
    assert r[r.metric=='auc_gt_2h'].value.dropna().between(0,1).all()
    for mode in ['unconditional','conditional','scrambled']:
        z=np.load(f'outputs/structure/features_{mode}.npz')
        assert np.array_equal(z['row_id'],np.arange(len(data)))
        position=z['logp_per_position'] if 'logp_per_position' in z else z['per_position']
        assert position.shape==(len(data),9)
        assert z['anchor_softmax'].shape==(len(data),40)
        assert z['template_order_scores'].shape==(15,len(data))
        assert np.allclose(z['mean_logp'],position.mean(1),atol=1e-6)
        assert np.allclose(z['anchor_softmax'].reshape(-1,2,20).sum(2),1,atol=1e-6)
        assert all(np.isfinite(z[k]).all() for k in ['mean_logp','anchor_softmax','spread','template_order_scores'])
    directories=['full','full_unconditional','full_conditional','full_unconditional_ranking','full_conditional_ranking']
    for d in directories:
        p=Path('outputs/baselines')/d
        assert (p/'integrity.json').exists(),p
        assert (p/'config.json').exists() and (p/'predictions.csv.gz').exists()
        integrity=json.loads((p/'integrity.json').read_text())
        assert not any(v is False for v in integrity.values()),p
    report=Path('report.html').read_text()
    assert 'Analysis in progress' not in report
    assert 'Replicate noise ceiling' in report or 'replicate noise ceiling' in report
    assert report.count('data:image/png;base64,')>=12
    for target in re.findall(r'href="([^"]+)"',report):
        if not target.startswith(('http:','https:','#','data:')):
            assert Path(target).exists(),target
    paths=[Path('report.html'),Path('results.csv'),Path('README.md'),Path('requirements.lock.txt')]
    paths+=list(Path('scripts').glob('*.py'))+list(Path('notes').glob('*.json'))
    paths+=list(Path('outputs').glob('*.csv'))
    paths+=list(Path('outputs/audit').glob('*'))
    paths+=list(Path('outputs/structure').glob('*.json'))+list(Path('outputs/structure').glob('features*.npz'))
    for d in directories: paths+=list((Path('outputs/baselines')/d).glob('*'))
    paths=[p for p in paths if p.is_file()]
    audit=json.loads(Path('outputs/audit/audit.json').read_text())
    assert sha(audit['source'])==audit['source_sha256']
    manifest={'data_sha256':audit['source_sha256'],'completed_model_loss_runs':len(expected),
              'metric_cells':len(r),'unavailable_metric_cells':int(r.value.isna().sum()),
              'all_required_runs_complete':True,'embedded_figures':report.count('data:image/png;base64,'),
              'checks':['metric grid and ranges','feature row alignment and normalization','15 ensemble members',
                        'saved supervised integrity reports','report completeness and local links','unchanged input data'],
              'files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(paths))}}
    Path('outputs/artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Final checks passed: {len(expected)} model/loss runs, {len(r)} metric cells, {manifest["embedded_figures"]} embedded figures.')

if __name__=='__main__':main()
