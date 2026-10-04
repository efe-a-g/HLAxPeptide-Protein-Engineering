"""Label-independent, persisted split manifests. Row ids are CSV row numbers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DATA = Path('resources/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv')
SEED = 20261003

def load_data():
    data = pd.read_csv(DATA)
    data.insert(0, 'row_id', np.arange(len(data)))
    return data

def balanced_subset(data, n=1000, seed=SEED):
    """Round-robin random sampling within allele; independent of labels."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(g.row_id.to_numpy()).tolist() for _, g in data.groupby('allele')]
    ids = []
    while len(ids) < min(n, len(data)):
        for group in groups:
            if group and len(ids) < min(n, len(data)):
                ids.append(group.pop())
    return data[data.row_id.isin(ids)].copy()

def make_splits(data, seed=SEED):
    ids = data.row_id.to_numpy()
    tr, te = train_test_split(ids, test_size=.2, random_state=seed, stratify=data.allele)
    yield 'random', 'holdout', tr, te
    i, j = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(data, groups=data.peptide))
    yield 'peptide_disjoint', 'holdout', ids[i], ids[j]
    for allele, group in data.groupby('allele', sort=True):
        yield 'leave_one_allele_out', allele, ids[~data.allele.eq(allele).to_numpy()], group.row_id.to_numpy()

def save_manifests(data, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    byid = data.set_index('row_id')
    for split, fold, tr, te in make_splits(data):
        train, test = byid.loc[tr], byid.loc[te]
        overlap = len(set(train.peptide) & set(test.peptide))
        if split == 'peptide_disjoint':
            assert overlap == 0
        if split == 'leave_one_allele_out':
            assert not set(train.allele) & set(test.allele)
        assert not set(tr) & set(te)
        # Test membership suffices: training = dataset universe minus test ids.
        rows.extend({'split': split, 'fold': fold, 'row_id': int(i)} for i in te)
        checks.append({'split': split, 'fold': fold, 'n_train': len(tr), 'n_test': len(te),
                       'peptides_in_both': overlap, 'test_rows_with_seen_peptide': int(test.peptide.isin(train.peptide).sum())})
    pd.DataFrame(rows).to_csv(outdir / 'test_membership.csv', index=False)
    pd.DataFrame(checks).to_csv(outdir / 'leakage_checks.csv', index=False)
    data[['row_id']].to_csv(outdir / 'universe.csv', index=False)
    (outdir / 'config.json').write_text(json.dumps({'seed': SEED, 'test_fraction': .2, 'training_membership': 'universe minus test rows', 'target_used_to_split': False}, indent=2))

if __name__ == '__main__':
    data = load_data()
    save_manifests(data, 'outputs/splits/full')
    pilot = balanced_subset(data)
    save_manifests(pilot, 'outputs/splits/pilot')
    pilot.to_csv('outputs/splits/pilot/data.csv', index=False)
    print(f'Saved full ({len(data)}) and allele-balanced pilot ({len(pilot)}) splits.')
