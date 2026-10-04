"""Reproducible LigandMPNN pMHC scoring with exact cached fixed-context decoding.

The unconditional mask hides ALL peptide identities while retaining preceding
HLA/beta2m identities. Unlike official ``use_sequence=False``, it is allele
conditioned. We cache fixed-context decoder states, then evaluate peptide nodes
only; tests compare this factorization with the original full decoder.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/structure'
VENDOR = ROOT / 'vendor/LigandMPNN'
sys.path.insert(0, str(VENDOR))
from model_utils import ProteinMPNN, cat_neighbors_nodes, gather_nodes
from splits import load_data, balanced_subset

AA = 'ACDEFGHIKLMNPQRSTVWY'
ATOM = ('N', 'CA', 'C', 'O')
TEMPLATES = {'1HHK': ('HLA-A*02:01','A','B','C'),
             '5IB1': ('HLA-B*27:05','A','B','C'),
             '7MLE': ('HLA-A*03:01','A','B','C'),
             '4NQX': ('HLA-A*01:01','A','B','M'),
             '3LKN': ('HLA-B*35:01','A','B','C')}
SEEDS = [17, 29, 43]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode_seq(s):
    return [AA.index(x) if x in AA else 20 for x in s]

def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)))

def build_template(pdb_id, assembly, data):
    """Use one spatially adjacent biological heterotrimer, not the crystal ASU."""
    allele,hla,b2m,pep = TEMPLATES[pdb_id]
    path = OUT/'pdb'/f'{pdb_id}.pdb'
    s = PDBParser(QUIET=True).get_structure(pdb_id,path)
    m = s[0]
    chains = [hla,pep] if assembly == 'truncated' else [hla,b2m,pep]
    seq,xyz,idx,chain_labels,hla_positions,hla_sequence_indices,pep_positions = [],[],[],[],[],[],[]
    residues_kept=[]
    missing=[]
    for chain_idx,c in enumerate(chains):
        for r in m[c]:
            if r.id[0] != ' ': continue
            resid = r.id[1]
            if c==hla and assembly=='truncated' and not 1 <= resid <= 182: continue
            if not all(a in r for a in ATOM):
                missing.append(f'{c}{resid}'); continue
            pos=len(seq)
            if c==hla and 1<=resid<=182:
                hla_positions.append(pos);hla_sequence_indices.append(resid-1)
            if c==pep: pep_positions.append(pos)
            seq.append(seq1(r.resname));xyz.append([r[a].coord.tolist() for a in ATOM])
            idx.append(resid);chain_labels.append(chain_idx);residues_kept.append((c,r))
    assert len(pep_positions)==9, (pdb_id,len(pep_positions))
    assert len(hla_positions)==182, (pdb_id,len(hla_positions),missing)
    native=''.join(seq[i] for i in hla_positions)
    matching=data.loc[data.allele==allele,'hla_seq'].iloc[0]
    mismatch=sum(a!=b for a,b in zip(native,matching))
    assert mismatch <= 2, (pdb_id,allele,mismatch)
    xyz=np.array(xyz,np.float32)
    hp=np.array(hla_positions); pp=np.array(pep_positions)
    distance=float(np.sqrt(((xyz[hp,1,None,:]-xyz[pp,1,:])**2).sum(-1)).min())
    assert distance < 6, (pdb_id,distance)
    # Save reviewable backbone-only PDBs preserving separate chain identifiers.
    backbone=OUT/'templates'/f'{pdb_id}_{assembly}.pdb';backbone.parent.mkdir(exist_ok=True)
    lines=[];serial=1;prev=None
    for c,r in residues_kept:
        if prev and prev!=c: lines.append('TER\n')
        for a in ATOM:
            x,y,z=r[a].coord
            lines.append(f'ATOM  {serial:5d} {a:^4s} {r.resname:3s} {c}{r.id[1]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {a[0]:>2s}\n');serial+=1
        prev=c
    backbone.write_text(''.join(lines)+'TER\nEND\n')
    n=len(seq)
    f={'X':torch.tensor(xyz)[None], 'S':torch.tensor(encode_seq(seq))[None],
       'mask':torch.ones(1,n),'R_idx':torch.tensor(idx)[None],
       'chain_labels':torch.tensor(chain_labels)[None],
       'Y':torch.zeros(1,n,1,3), 'Y_t':torch.zeros(1,n,1,dtype=torch.long),
       'Y_m':torch.zeros(1,n,1),'chain_mask':torch.zeros(1,n),
       'batch_size':1,'symmetry_residues':[[]]}
    f['chain_mask'][:,pp]=1
    meta={'pdb_id':pdb_id,'native_allele':allele,'resolution':s.header.get('resolution'),
          'native_peptide':''.join(seq[i] for i in pp),'assembly':assembly,
          'residues':n,'hla_domain_residues':len(hp),'peptide_residues':9,
          'closest_hla_peptide_ca_distance':distance,'native_hla_mismatches':mismatch,
          'missing_backbone_residues':missing,'pdb_sha256':digest(path),
          'template_sha256':digest(backbone),'source':f'https://www.rcsb.org/structure/{pdb_id}'}
    return f,hp,np.array(hla_sequence_indices),pp,meta

class Scorer:
    def __init__(self, threads=2):
        torch.set_num_threads(threads);torch.manual_seed(2026)
        checkpoint=VENDOR/'model_params/ligandmpnn_v_32_010_25.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=ProteinMPNN(k_neighbors=ck['num_edges'],atom_context_num=ck['atom_context_num'],
            model_type='ligand_mpnn',device=torch.device('cpu')).eval()
        self.model.load_state_dict(ck['model_state_dict'])
        self.provenance={'checkpoint':checkpoint.name,'checkpoint_sha256':digest(checkpoint),
                         'source_commit':subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),
                         'pipeline_sha256':digest(__file__),'model':'LigandMPNN','atom_context':False,
                         'side_chain_context':False,'device':'cpu','threads':threads,
                         'unconditional_mode':'all peptide identities hidden; fixed HLA sequence retained',
                         'conditional_mode':'autoregressive; HLA fixed first, random peptide orders'}
        self.provenance['scoring_algorithm_sha256']=hashlib.sha256(
            (inspect.getsource(type(self).prepare)+inspect.getsource(type(self).score)+
             inspect.getsource(build_template)+digest(VENDOR/'model_utils.py')).encode()).hexdigest()

    @torch.inference_mode()
    def prepare(self, template, hla_sequence, seed, scrambled=False):
        f,hp,hsi,pp,meta=template
        m=self.model
        s=f['S'].clone()
        hla=encode_seq(hla_sequence)
        if scrambled: hla=np.random.default_rng(seed+701).permutation(hla).tolist()
        s[:,hp]=torch.tensor(hla)[hsi]
        # Encoder depends only on backbone; compute once for this assembly/template.
        if '_encoded' not in f: f['_encoded']=m.encode(f)
        hv,he,ei=f['_encoded'];n=s.shape[1]
        rng=np.random.default_rng(seed)
        fixed=np.setdiff1d(np.arange(n),pp)
        order=np.concatenate((rng.permutation(fixed),pp))
        ranks=torch.tensor(np.argsort(order))[None]
        neighbor_rank=gather_nodes(ranks[:,:,None],ei).squeeze(-1)
        bw=(neighbor_rank<ranks[:,:,None]).float()[:,:,:,None]
        hs=m.W_s(s);hes=cat_neighbors_nodes(hs,he,ei)
        hexv=cat_neighbors_nodes(hv,cat_neighbors_nodes(torch.zeros_like(hs),he,ei),ei)
        states=[hv]
        for layer in m.decoder_layers:
            hv=layer(hv,bw*cat_neighbors_nodes(hv,hes,ei)+(1-bw)*hexv,f['mask'])
            states.append(hv)
        # Fixed node states cannot see any peptide sequence because all precede it.
        pe=ei[:,pp];lookup=torch.full((n,),-1,dtype=torch.long);lookup[pp]=torch.arange(9)
        neighbor_pep=lookup[pe][0]
        return {'pp':pp,'f':f,'s':s,'ranks':ranks,'seed':seed,'neighbor_pep':neighbor_pep,
                'ei':pe,'he':he[:,pp], 'hs_fixed':gather_nodes(hs,pe),
                'hv_fixed':[gather_nodes(x,pe) for x in states[:-1]],
                'hv0':states[0][:,pp], 'hexv':hexv[:,pp]}

    @torch.inference_mode()
    def score(self, context, peptides, mode):
        m=self.model;c=context
        x=torch.tensor(np.array([encode_seq(s) for s in peptides]),dtype=torch.long)
        b=len(peptides);j=c['neighbor_pep'];is_pep=j>=0;safe=j.clamp(min=0)
        hs=m.W_s(x)
        hs_n=torch.where(is_pep[None,:,:,None],hs[:,safe],c['hs_fixed'])
        hes=torch.cat((c['he'].expand(b,-1,-1,-1),hs_n),-1)
        if mode=='unconditional': attend=~is_pep
        else:
            # Stable random order per ensemble seed; shared across rows to reduce noise.
            rank=np.argsort(np.random.default_rng(c['seed']+9001).permutation(9))
            rank=torch.tensor(rank)
            attend=(~is_pep)|(rank[safe]<rank[:,None])
        bw=attend.float()[None,:,:,None]
        enc=(1-bw)*c['hexv']
        hv=c['hv0'].expand(b,-1,-1)
        for k,layer in enumerate(m.decoder_layers):
            hv_n=torch.where(is_pep[None,:,:,None],hv[:,safe],c['hv_fixed'][k])
            hesv=torch.cat((hes,hv_n),-1)
            hv=layer(hv,bw*hesv+enc,torch.ones(b,9))
        return torch.log_softmax(m.W_out(hv),-1)[:,:,:20].cpu().numpy()

    @torch.inference_mode()
    def validate(self, template, sequence):
        c=self.prepare(template,sequence,SEEDS[0]);f,hp,hsi,pp,meta=template
        peptides=['ACDEFGHIK','LMNPQRSTV']
        a=self.score(c,peptides,'unconditional')
        assert np.max(np.abs(a[0]-a[1])) < 2e-6
        # Full official autoregressive score with exactly matching fixed and peptide orders.
        local={k:v for k,v in f.items() if not k.startswith('_')}
        local['S']=c['s'].clone();local['S'][:,pp]=torch.tensor(encode_seq(peptides[0]))
        ranks=c['ranks'].float()+1
        rank=np.argsort(np.random.default_rng(SEEDS[0]+9001).permutation(9))
        ranks[:,pp]=torch.tensor(rank+len(ranks[0])-9+1,dtype=torch.float32)
        local['randn']=ranks/(local['chain_mask']+.0001)
        official=self.model.score(local,True)['log_probs'][:,pp,:20].cpu().numpy()
        cached=self.score(c,[peptides[0]],'conditional')
        delta=float(np.max(np.abs(official-cached)));assert delta<3e-5,delta
        # For unconditional, every peptide independently has order immediately after HLA.
        reference=[]
        for target in range(9):
            rank_target=np.arange(9)+1;rank_target[target]=0
            ranks[:,pp]=torch.tensor(rank_target+len(ranks[0])-9+1,dtype=torch.float32)
            local['randn']=ranks/(local['chain_mask']+.0001)
            reference.append(self.model.score(local,True)['log_probs'][0,pp[target],:20].cpu().numpy())
        u_delta=float(np.max(np.abs(np.array(reference)-a[0])));assert u_delta<3e-5,u_delta
        scrambled=self.score(self.prepare(template,sequence,SEEDS[0],True),peptides,'unconditional')
        hla_change=float(np.max(np.abs(a-scrambled)));assert hla_change>1e-4
        return {'conditional_official_max_abs_error':delta,'unconditional_nine_official_calls_max_abs_error':u_delta,
                'unconditional_peptide_mutation_max_abs_error':float(np.max(np.abs(a[0]-a[1]))),
                'unconditional_hla_shuffle_max_abs_change':hla_change,'passed':True}

def features_from_logps(data, logps):
    """logps is [ensemble,rows,9,20], average log-likelihood not log of mean p."""
    x=np.array([encode_seq(p) for p in data.peptide])
    taken=np.take_along_axis(logps,x[None,:,:,None],axis=-1)[...,0]
    per=taken.mean(0)
    p=np.exp(logps);p=p/p.sum(-1,keepdims=True)
    anchor=p[:,:, [1,8],:].mean(0).reshape(len(data),40)
    return {'row_id':data.row_id.to_numpy(),'mean_logp':per.mean(-1),'per_position':per,'logp_per_position':per,
            'anchor_softmax':anchor,'spread':taken.mean(-1).std(0),
            'template_order_scores':taken.mean(-1)}

def cache_path(scorer,meta,allele,hla,mode,seed,scrambled=False,peptides=None):
    key={'checkpoint':scorer.provenance['checkpoint_sha256'],'source_commit':scorer.provenance['source_commit'],
         'scoring_algorithm_sha256':scorer.provenance['scoring_algorithm_sha256'],'template':meta['template_sha256'],
         'allele':allele,'hla':hla,'mode':mode,'seed':seed,'scrambled':scrambled,
         'peptides':peptides}
    # Conditional archives index peptide sequences internally; cache identity is
    # (this context key, peptide). Growing a subset reuses existing peptide rows.
    h=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    p=OUT/'cache'/f'{h}.npz';p.parent.mkdir(exist_ok=True)
    return p,key

def compute(scorer,data,templates,mode,scrambled=False):
    ensembles=[];motifs=[];cache_manifest=[]
    t0=time.time()
    for template in templates:
        f,hp,hsi,pp,meta=template
        for seed in SEEDS:
            arr=np.empty((len(data),9,20),np.float32)
            motif={}
            for allele,g in data.groupby('allele',sort=True):
                positions=data.index.get_indexer(g.index)
                hla=g.hla_seq.iloc[0]
                peptides=None if mode=='unconditional' else g.peptide.tolist()
                path,key=cache_path(scorer,meta,allele,hla,mode,seed,scrambled,None)
                if mode=='conditional':
                    if path.exists():
                        archive=np.load(path);old_peptides=archive['peptide'].tolist();old_lp=archive['log_probs']
                    else: old_peptides=[];old_lp=np.empty((0,9,20),np.float32)
                    lookup={p:i for i,p in enumerate(old_peptides)}
                    missing=list(dict.fromkeys(p for p in peptides if p not in lookup))
                    if missing:
                        c=scorer.prepare(template,hla,seed,scrambled)
                        new_lp=np.concatenate([scorer.score(c,missing[i:i+64],mode) for i in range(0,len(missing),64)])
                        old_lp=np.concatenate((old_lp,new_lp));old_peptides+=missing
                        lookup={p:i for i,p in enumerate(old_peptides)}
                        np.savez_compressed(path,log_probs=old_lp,peptide=np.array(old_peptides),key=json.dumps(key,sort_keys=True))
                    lp=old_lp[[lookup[p] for p in peptides]]
                elif path.exists(): lp=np.load(path)['log_probs']
                else:
                    c=scorer.prepare(template,hla,seed,scrambled)
                    lp=scorer.score(c,['XXXXXXXXX'],mode)[0]
                    np.savez_compressed(path,log_probs=lp,key=json.dumps(key,sort_keys=True))
                if mode=='unconditional':
                    motif[allele]=lp
                    arr[positions]=lp
                else: arr[positions]=lp
                cache_manifest.append({'path':str(path.relative_to(ROOT)),**key})
            ensembles.append(arr);motifs.append(motif)
            print(f'{mode} scramble={scrambled} {meta["pdb_id"]} {meta["assembly"]} seed={seed}: {time.time()-t0:.1f}s',flush=True)
    return np.stack(ensembles),motifs,cache_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','pilot','full','controls'],default='validate')
    p.add_argument('--mode',choices=['unconditional','conditional','both'],default='both')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--assembly',choices=['truncated','full'],default='truncated')
    args=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    data=load_data();scorer=Scorer(args.threads)
    templates=[build_template(t,'truncated',data) for t in TEMPLATES]
    full=[build_template(t,'full',data) for t in TEMPLATES]
    dump_json(OUT/'templates.json',[x[-1] for x in templates+full])
    dump_json(OUT/'provenance.json',scorer.provenance)
    validation=scorer.validate(templates[0],data.hla_seq.iloc[0])
    dump_json(OUT/'decoder_validation.json',validation);print(validation,flush=True)
    if args.stage=='validate': return
    if args.stage=='controls':
        subset=balanced_subset(data,500)
        subset.to_csv(OUT/'control500_rows.csv',index=False)
        for assembly,ts in [('truncated',templates),('full',full)]:
            lp,_,manifest=compute(scorer,subset,ts,'unconditional')
            np.savez_compressed(OUT/f'control500_{assembly}.npz',**features_from_logps(subset,lp))
            dump_json(OUT/f'cache_manifest_control500_{assembly}.json',manifest)
        return
    templates=templates if args.assembly=='truncated' else full
    subset=balanced_subset(data,1000) if args.stage=='pilot' else data
    suffix='_pilot' if args.stage=='pilot' else ''
    subset.to_csv(OUT/f'rows{suffix}.csv',index=False)
    for mode in (['unconditional','conditional'] if args.mode=='both' else [args.mode]):
        lp,motifs,manifest=compute(scorer,subset,templates,mode)
        np.savez_compressed(OUT/f'features_{mode}{suffix}.npz',**features_from_logps(subset,lp))
        dump_json(OUT/f'cache_manifest_{mode}{suffix}.json',manifest)
        if mode=='unconditional':
            alleles=sorted(subset.allele.unique())
            mat=np.array([[m[a] for a in alleles] for m in motifs])
            np.savez_compressed(OUT/f'unconditional_motifs{suffix}.npz',alleles=alleles,log_probs=mat,
                                templates=np.repeat(list(TEMPLATES),len(SEEDS)),seeds=np.tile(SEEDS,len(TEMPLATES)))
            scrambled,_,manifest=compute(scorer,subset,templates,mode,True)
            np.savez_compressed(OUT/f'features_scrambled{suffix}.npz',**features_from_logps(subset,scrambled))
            dump_json(OUT/f'cache_manifest_scrambled{suffix}.json',manifest)
    dump_json(OUT/f'run_{args.stage}_{args.mode}.json',{'args':vars(args),'rows':len(subset),
               'seeds':SEEDS,'templates':list(TEMPLATES),'assembly':args.assembly,
               'provenance':scorer.provenance,'complete':True})

if __name__=='__main__': main()
