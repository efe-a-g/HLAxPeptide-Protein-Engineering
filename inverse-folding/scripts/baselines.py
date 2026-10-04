"""Fold-local supervised baseline ladder with fixed training budgets.

Example: .venv/bin/python scripts/baselines.py --stage pilot
No test target enters feature scaling, motifs, training, or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from torch import nn
from torch.nn import functional as F

from evaluation import EPSILON, evaluate_predictions
from splits import SEED, balanced_subset, load_data, make_splits

AA = 'ACDEFGHIKLMNPQRSTVWY'


def stable_seed(*parts):
    value = '|'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little')


def representations(data):
    aa_index = {a: i for i, a in enumerate(AA)}
    peptides = np.asarray([[aa_index[a] for a in p] for p in data.peptide], dtype=np.int64)
    pseudo = np.asarray([[aa_index[a] for a in p] for p in data.hla_pseudoseq], dtype=np.int64)
    matrix = substitution_matrices.load('BLOSUM50')
    blosum = np.asarray([[matrix[a, b] / 5. for b in AA] for a in AA], dtype=np.float32)
    peptide_features = blosum[peptides].reshape(len(data), -1)
    pseudoseq_features = np.eye(20, dtype=np.float32)[pseudo].reshape(len(data), -1)
    allele_names = sorted(data.allele.unique())
    allele_mapping = {a: i for i, a in enumerate(allele_names)}
    alleles = np.asarray([allele_mapping[a] for a in data.allele], dtype=np.int64)
    return peptides, peptide_features, pseudoseq_features, alleles, allele_names


class Head(nn.Module):
    def __init__(self, n_features, n_alleles=None, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_alleles, embedding_dim) if n_alleles else None
        self.layers = nn.Sequential(
            nn.Linear(n_features + (embedding_dim if n_alleles else 0), 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, alleles):
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(alleles)], dim=-1)
        return self.layers(x).squeeze(-1)


def ranking_pairs(alleles, targets, generator):
    """Two random within-allele matchings per minibatch; ties are discarded."""
    left, right = [], []
    for a in torch.unique(alleles):
        ix = torch.nonzero(alleles == a, as_tuple=True)[0]
        if len(ix) < 2:
            continue
        for _ in range(2):
            other = ix[torch.randperm(len(ix), generator=generator)]
            keep = targets[ix] != targets[other]
            left.append(ix[keep])
            right.append(other[keep])
    if not left:
        return None
    return torch.cat(left), torch.cat(right)


def train_predict(features, alleles, target, train, test, loss, seed, args, n_alleles=None):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    # The same representation and fold initialize identically across loss ablations.
    model = Head(features.shape[1], n_alleles)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    a = torch.from_numpy(alleles)
    t = torch.from_numpy(np.asarray(target, dtype=np.float32))
    tr = torch.from_numpy(train)
    y = torch.log(t + EPSILON)
    floor = math.log(args.censor_threshold + EPSILON)
    last_loss = np.nan
    steps = 0
    start = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        shuffled = tr[torch.randperm(len(tr), generator=generator)]
        for ix in shuffled.split(args.batch_size):
            pred = model(x[ix], a[ix])
            if loss == 'mse':
                objective = F.mse_loss(pred, y[ix])
            elif loss == 'censored_hinge':
                error = torch.where(t[ix] > 0, pred-y[ix], F.relu(pred-floor))
                objective = error.square().mean()
            elif loss == 'tobit':
                # Fixed sigma=1 on log-hour scale: CDF term for assumed zeros.
                terms = torch.where(t[ix] > 0, .5*(pred-y[ix]).square(),
                                    -torch.special.log_ndtr(floor-pred))
                objective = terms.mean()
            elif loss == 'ranking':
                pairs = ranking_pairs(a[ix], t[ix], generator)
                if pairs is None or len(pairs[0]) == 0:
                    continue
                left, right = pairs
                direction = torch.sign(t[ix][left]-t[ix][right])
                objective = F.softplus(-direction*(pred[left]-pred[right])).mean()
            else:
                raise ValueError(loss)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            last_loss = float(objective.detach())
            steps += 1
    model.eval()
    with torch.no_grad():
        prediction = torch.cat([model(x[ix], a[ix]) for ix in torch.from_numpy(test).split(2048)]).numpy()
    return prediction, {'seconds': time.perf_counter()-start, 'last_batch_loss': last_loss,
                        'optimizer_steps': steps, 'n_parameters': sum(p.numel() for p in model.parameters())}


def motif_predict(peptides, alleles, target, train, test):
    """Train-only stable (>2h) PWM, one background pseudocount per residue."""
    background = np.bincount(peptides[train].ravel(), minlength=20).astype(float) + 1.
    background /= background.sum()
    pred = np.full(len(test), np.nan)
    status = np.full(len(test), 'unseen_allele', dtype=object)
    for a in np.unique(alleles[test]):
        tr = train[alleles[train] == a]
        if len(tr) == 0:
            continue
        stable = tr[target[tr] > 2]
        if len(stable) == 0:
            status[alleles[test] == a] = 'no_stable_train_peptides'
            continue
        counts = np.stack([np.bincount(peptides[stable, i], minlength=20) for i in range(9)]).astype(float)
        pwm = (counts + 20*background[None, :]) / (len(stable)+20)
        logodds = np.log(pwm/background[None, :])
        mask = alleles[test] == a
        pred[mask] = logodds[np.arange(9)[None, :], peptides[test[mask]]].sum(axis=1)
        status[mask] = 'ok'
    return pred, status


def read_structural(path, data):
    """NPZ contract: row_id, mean_logp[N,1], logp_per_position[N,9],
    anchor_softmax[N,40]. Per-mode files can be evaluated independently.
    """
    archive = np.load(path)
    index = {int(v): i for i, v in enumerate(archive['row_id'])}
    order = np.asarray([index[int(v)] for v in data.row_id])
    mode = 'unconditional' if 'unconditional' in Path(path).name else 'conditional' if 'conditional' in Path(path).name else 'structural'
    features = {}
    for name in ['mean_logp', 'logp_per_position', 'anchor_softmax']:
        key = 'per_position' if name == 'logp_per_position' and 'per_position' in archive else name
        value = archive[key][order].reshape(len(data), -1).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite structural feature: {name}')
        features['baseline_plus_'+mode+'_'+name] = value
    features['baseline_plus_'+mode+'_all'] = np.concatenate(list(features.values()), axis=1)
    return features


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    np.random.seed(SEED)
    data = load_data()
    if args.stage == 'pilot':
        data = balanced_subset(data)
    data = data.reset_index(drop=True)
    outdir = Path(args.output or f'outputs/baselines/{args.stage}')
    outdir.mkdir(parents=True, exist_ok=True)
    row_to_pos = {v: i for i, v in enumerate(data.row_id)}
    peptides, peptide_x, pseudo_x, alleles, allele_names = representations(data)
    baseline_x = np.concatenate([peptide_x, pseudo_x], axis=1)
    target = data.thalf_hours.to_numpy()
    structural = read_structural(args.structural, data) if args.structural else {}
    if structural:
        suffixes = args.structural_features.split(',')
        structural = {k: v for k, v in structural.items() if any(k.endswith('_'+s) for s in suffixes)}
        if not structural:
            raise ValueError('No structural features selected')
    requested = [r for r in args.representations.split(',') if r]
    if args.structural:
        requested += list(structural)
    requested = list(dict.fromkeys(requested))
    losses = args.losses.split(',')
    config = vars(args) | {'seed': SEED, 'n_rows': len(data), 'n_alleles': len(allele_names),
                           'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'epsilon': EPSILON, 'amino_acid_order': AA,
                           'peptide_encoding': 'BLOSUM50 / 5, flattened 9x20',
                           'pseudo_encoding': 'one-hot 34x20', 'embedding_dim': 32,
                           'hidden_widths': [64,32], 'torch_threads': 2,
                           'primary_loss': 'mse', 'primary_split': 'leave_one_allele_out',
                           'primary_augmentation': 'all structural features combined',
                           'component_ablations': ['mean_logp','logp_per_position','anchor_softmax'],
                           'experiment_note': ('Exploratory follow-up after baseline loss comparison; original MSE primary comparison unchanged.'
                                               if args.exploratory else 'Prespecified representation and loss comparison.'),
                           'censoring_note': 'Assumed zero censoring; c=0.1h is sensitivity assumption, not verified detection floor.',
                           'ranking_note': 'Two random matchings per allele per batch, exact target ties skipped.',
                           'target_note': 'No test-target use or test-driven early stopping; fixed epoch budget.',
                           'pwm_note': 'Train half-life >2h only; 20 train-background pseudocounts per position.'}
    (outdir/'config.json').write_text(json.dumps(config, indent=2))
    pieces, timings = [], []
    existing = outdir/'predictions.csv.gz'
    if args.resume and existing.exists():
        pieces.append(pd.read_csv(existing))
    done = set()
    if pieces:
        done = set(map(tuple, pieces[0][['split','fold','representation','loss']].drop_duplicates().to_numpy()))
    folds = list(make_splits(data))
    if args.splits:
        folds = [f for f in folds if f[0] in args.splits.split(',')]
    if args.max_folds:
        folds = folds[:args.max_folds]
    for fi, (split, fold, train_ids, test_ids) in enumerate(folds):
        train = np.asarray([row_to_pos[i] for i in train_ids], dtype=np.int64)
        test = np.asarray([row_to_pos[i] for i in test_ids], dtype=np.int64)
        base = data.iloc[test][['row_id','allele','thalf_hours']].copy()
        base['split'], base['fold'] = split, fold
        for representation in requested:
            rep_losses = ['reference'] if representation in ['allele_mean','pwm'] else losses
            for loss in rep_losses:
                key = (split,fold,representation,loss)
                if key in done:
                    continue
                stats, status = {}, np.full(len(test), 'ok', dtype=object)
                if representation == 'allele_mean':
                    logtarget = np.log(target+EPSILON)
                    overall = logtarget[train].mean()
                    means = {a: logtarget[train[alleles[train] == a]].mean() for a in np.unique(alleles[train])}
                    pred = np.asarray([means.get(a, overall) for a in alleles[test]])
                    status = np.asarray(['train_allele_mean' if a in means else 'global_train_mean_fallback' for a in alleles[test]])
                elif representation == 'pwm':
                    pred, status = motif_predict(peptides, alleles, target, train, test)
                elif representation == 'allele_embedding' and split == 'leave_one_allele_out':
                    pred = np.full(len(test), np.nan)
                    status[:] = 'unsupported_unseen_categorical_allele'
                else:
                    if representation == 'allele_embedding':
                        features, n_alleles = peptide_x, len(allele_names)
                    elif representation == 'blosum_pseudoseq':
                        features, n_alleles = baseline_x, None
                    elif representation in structural:
                        value = structural[representation]
                        mean, std = value[train].mean(0), value[train].std(0)
                        std = np.where(std > 1e-6, std, 1.)
                        features = np.concatenate([baseline_x, (value-mean)/std], axis=1)
                        n_alleles = None
                    else:
                        raise ValueError(f'Unknown representation {representation}')
                    pred, stats = train_predict(features, alleles, target, train, test, loss,
                                                stable_seed(SEED, split, fold, representation), args, n_alleles)
                    # Seen-allele categorical training must never treat an untrained embedding as valid.
                    if representation == 'allele_embedding':
                        unseen = ~np.isin(alleles[test], alleles[train])
                        pred[unseen] = np.nan
                        status[unseen] = 'unsupported_unseen_categorical_allele'
                frame = base.copy()
                frame['representation'], frame['loss'] = representation, loss
                frame['prediction'], frame['prediction_status'] = pred, status
                pieces.append(frame)
                timings.append({'split':split,'fold':fold,'representation':representation,'loss':loss,**stats})
                print(json.dumps({'fold_index':fi+1,'n_folds':len(folds),**timings[-1]}), flush=True)
        # Persist each fold so long runs remain inspectable and recoverable.
        merged = pd.concat(pieces, ignore_index=True)
        pending = outdir/'.predictions.csv.gz.tmp'
        merged.to_csv(pending, index=False, compression={'method':'gzip', 'compresslevel':1, 'mtime':0})
        pending.replace(existing)
        pd.DataFrame(timings).to_csv(outdir/'timings.csv', index=False)
    merged = pd.concat(pieces, ignore_index=True)
    evaluate_predictions(merged, outdir)
    print(f'Completed {args.stage}: {len(merged)} prediction rows saved to {existing}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot','full'], default='pilot')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--censor-threshold', type=float, default=.1)
    parser.add_argument('--losses', default='mse,censored_hinge,ranking')
    parser.add_argument('--representations', default='allele_mean,pwm,allele_embedding,blosum_pseudoseq')
    parser.add_argument('--structural', help='NPZ with row-aligned structural features; see read_structural.')
    parser.add_argument('--structural-features', default='mean_logp,logp_per_position,anchor_softmax,all',
                        help='Comma-separated augmentation components; use all for combined features only.')
    parser.add_argument('--exploratory', action='store_true', help='Record a follow-up analysis rather than original primary comparison.')
    parser.add_argument('--splits', help='Optional comma-separated subset of split names.')
    parser.add_argument('--max-folds', type=int)
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
