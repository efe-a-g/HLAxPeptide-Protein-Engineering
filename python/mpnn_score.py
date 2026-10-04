#!/usr/bin/env python3
"""
Inverse-folding (ProteinMPNN/LigandMPNN) scoring of peptide-HLA class I complexes.

An inverse-folding model gives p(sequence | backbone). Conditioning on the groove geometry and
scoring the observed peptide is a zero-shot stability proxy that uses the conserved clamp and
pocket geometry the 34-aa pseudosequence discards. No training and no labels, so it is immune
to every split concern and is pan-specific by construction.

Threading. Every `hla_seq` in this dataset is exactly 182 residues and starts GSHSMRY..., the
mature class I heavy chain, so alpha1/alpha2 maps 1:1 onto residues 1-182 of any template
heavy chain with no alignment step. We keep one template backbone and swap in each allele's
sequence, which is what makes 75 alleles affordable.

Two scoring modes, which are different experiments:

  (b) UNCONDITIONAL  the peptide sequence is NOT an input. For each peptide position i,
      p(x_i | backbone, hla_seq): the geometry's own statement of what the groove wants at i,
      independent of the other peptide positions. Gives a 9x20 matrix per allele; score any
      peptide by lookup and sum. One batched pass per allele.

  (a) CONDITIONAL    the peptide sequence IS an input. sum_i log p(x_i | x_<i, backbone,
      hla_seq), so inter-position dependence is captured.

Decoding order is what separates them. `score()` builds it as argsort((chain_mask + 1e-4) *
|randn|), so with chain_mask = 1 on the peptide and 0 on the HLA, the HLA is always decoded
first and the peptide last; `randn` then orders the peptide positions among themselves. For
(b) we craft `randn` so that batch element k decodes peptide position k first -- that position
then sees the HLA sequence and no other peptide residue, which is exactly p(x_i | backbone,
hla_seq). Note this is NOT what `use_sequence=False` does: that flag drops sequence context
everywhere, including the HLA, which is a different quantity.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

MPNN_DIR = os.environ.get("MPNN_DIR", "/tmp/LigandMPNN")
sys.path.insert(0, MPNN_DIR)
from data_utils import alphabet as MPNN_ALPHABET  # noqa: E402
from data_utils import featurize, parse_PDB  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402

AA20 = list("ARNDCQEGHILKMFPSTWYV")
MPNN_IDX = {a: MPNN_ALPHABET.index(a) for a in AA20}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3,
        num_decoder_layers=3, k_neighbors=ckpt["num_edges"], device=device,
        atom_context_num=1, model_type="protein_mpnn", ligand_mpnn_use_side_chain_context=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_template(pdb_path, hla_chain, pep_chain, n_hla=182, n_pep=9, device="cpu"):
    """Heavy-chain residues 1..n_hla plus the whole peptide chain, in that order."""
    pd_, _ = parse_PDB(pdb_path, device=device, chains=[hla_chain, pep_chain])[:2]
    letters = np.asarray(pd_["chain_letters"])
    rid = pd_["R_idx"].cpu().numpy()

    hla_sel = np.flatnonzero((letters == hla_chain) & (rid >= 1) & (rid <= n_hla))
    hla_sel = hla_sel[np.argsort(rid[hla_sel])]
    pep_sel = np.flatnonzero(letters == pep_chain)
    pep_sel = pep_sel[np.argsort(rid[pep_sel])]
    if len(hla_sel) != n_hla or len(pep_sel) != n_pep:
        raise ValueError(f"{os.path.basename(pdb_path)}: got {len(hla_sel)} HLA residues "
                         f"(want {n_hla}) and {len(pep_sel)} peptide (want {n_pep}); "
                         f"HLA gaps at {sorted(set(range(1, n_hla + 1)) - set(rid[hla_sel]))[:12]}")
    keep = np.concatenate([hla_sel, pep_sel])

    out = {}
    for k in ("X", "mask", "R_idx", "chain_labels", "S", "xyz_37", "xyz_37_m"):
        if k in pd_:
            out[k] = pd_[k][keep]
    out["chain_letters"] = letters[keep]
    # Renumber so the two chains do not collide in the positional encoding.
    out["R_idx"] = torch.cat([torch.arange(1, n_hla + 1), torch.arange(1, n_pep + 1) + 1000]
                             ).to(out["X"].device)
    out["chain_labels"] = torch.cat([torch.zeros(n_hla), torch.ones(n_pep)]).to(out["X"].device)
    # Peptide designable, HLA fixed context -> HLA always decoded first.
    out["chain_mask"] = torch.cat([torch.zeros(n_hla), torch.ones(n_pep)]).to(out["X"].device)
    out["_n_hla"], out["_n_pep"] = n_hla, n_pep
    return out


def thread(tpl, hla_seq, pep_seq=None):
    """Write an allele's sequence onto the template backbone; peptide optional."""
    n_hla, n_pep = tpl["_n_hla"], tpl["_n_pep"]
    S = tpl["S"].clone()
    S[:n_hla] = torch.tensor([MPNN_IDX[a] for a in hla_seq], device=S.device)
    if pep_seq is not None:
        S[n_hla:] = torch.tensor([MPNN_IDX[a] for a in pep_seq], device=S.device)
    d = {k: v for k, v in tpl.items() if not k.startswith("_")}
    d["S"] = S
    return d


def _fd(protein_dict, device, batch_size):
    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=False,
                   number_of_ligand_atoms=1, model_type="protein_mpnn")
    fd["batch_size"] = batch_size
    fd["symmetry_residues"] = [[]]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fd.items()}


@torch.no_grad()
def unconditional_pssm(model, tpl, hla_seq, device, n_hla=182, n_pep=9):
    """(b) 9x20 matrix: log p(x_i | backbone, hla_seq), independent across peptide positions."""
    fd = _fd(thread(tpl, hla_seq), device, batch_size=n_pep)
    L = fd["mask"].shape[1]
    # Batch element k gives peptide position k the smallest |randn| in the peptide block, so k
    # is decoded first among the peptide and sees only the HLA sequence.
    randn = torch.ones(n_pep, L, device=device)
    randn[:, n_hla:] = 2.0
    for k in range(n_pep):
        randn[k, n_hla + k] = 0.01
    fd["randn"] = randn
    out = model.score(fd, use_sequence=True)
    lp = out["log_probs"]                                   # [n_pep, L, 21]
    pssm = torch.stack([lp[k, n_hla + k] for k in range(n_pep)])     # [9, 21]
    return pssm[:, [MPNN_IDX[a] for a in AA20]].cpu().numpy()        # [9, 20] in AA20 order


@torch.no_grad()
def conditional_scores(model, tpl, hla_seq, peptides, device, n_orders=5, seed=0,
                       batch=256, n_hla=182, n_pep=9):
    """(a) sum_i log p(x_i | x_<i, backbone, hla_seq), averaged over random decoding orders.

    Returns (mean_logp [N], per_position_logp [N, 9]); both averaged over `n_orders` orders.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    orders = [torch.cat([torch.ones(n_hla), 2.0 + torch.rand(n_pep, generator=g)]).to(device)
              for _ in range(n_orders)]
    tot = np.zeros((len(peptides), n_pep), dtype=np.float64)
    pep_idx = np.array([[MPNN_IDX[a] for a in p] for p in peptides])
    for start in range(0, len(peptides), batch):
        chunk = peptides[start:start + batch]
        B = len(chunk)
        fd = _fd(thread(tpl, hla_seq), device, batch_size=B)
        S = fd["S"].repeat(B, 1)
        S[:, n_hla:] = torch.tensor(pep_idx[start:start + B], device=device)
        fd["S"] = S
        fd["mask"] = fd["mask"].repeat(B, 1)
        fd["chain_mask"] = fd["chain_mask"].repeat(B, 1)
        acc = np.zeros((B, n_pep))
        for o in orders:
            fd["randn"] = o.unsqueeze(0).repeat(B, 1)
            lp = model.score(fd, use_sequence=True)["log_probs"][:, n_hla:, :]
            acc += lp.gather(2, torch.tensor(pep_idx[start:start + B], device=device
                                             ).unsqueeze(-1)).squeeze(-1).cpu().numpy()
        tot[start:start + B] = acc / len(orders)
    return tot.sum(1), tot


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data_path",
                   default="data/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv")
    p.add_argument("--templates", required=True,
                   help="JSON list of {pdb, hla_chain, pep_chain, allele, name}")
    p.add_argument("--checkpoint", default=f"{MPNN_DIR}/model_params/proteinmpnn_v_48_020.pt")
    p.add_argument("--mode", default="uncond", choices=["uncond", "cond"])
    p.add_argument("--n_orders", type=int, default=5)
    p.add_argument("--scramble", action="store_true",
                   help="control: shuffle hla_seq before scoring")
    p.add_argument("--subset", type=int, default=0, help="limit rows (cond mode)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="outputs/mpnn")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    templates = json.load(open(args.templates))
    df = pd.read_csv(args.data_path)
    alleles = df.drop_duplicates("allele")[["allele", "hla_seq"]].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    if args.mode == "uncond":
        rows = []
        for t in templates:
            tpl = load_template(t["pdb"], t["hla_chain"], t["pep_chain"], device=args.device)
            print(f"[*] template {t['name']} ({t.get('allele','?')}) loaded", flush=True)
            for _, r in alleles.iterrows():
                seq = r.hla_seq
                if args.scramble:
                    seq = "".join(rng.permutation(list(seq)))
                pssm = unconditional_pssm(model, tpl, seq, device)
                rows.append({"template": t["name"], "allele": r.allele, "pssm": pssm})
            print(f"    scored {len(alleles)} alleles", flush=True)
        tag = "_scrambled" if args.scramble else ""
        np.savez(os.path.join(args.output_dir, f"uncond_pssm{tag}.npz"),
                 pssm=np.stack([r["pssm"] for r in rows]),
                 template=np.array([r["template"] for r in rows]),
                 allele=np.array([r["allele"] for r in rows]), aa_order=np.array(AA20))
        print(f"[*] wrote uncond_pssm{tag}.npz  ({len(rows)} allele-template pairs)")
    else:
        d = df if not args.subset else df.sample(args.subset, random_state=args.seed)
        out = []
        for t in templates:
            tpl = load_template(t["pdb"], t["hla_chain"], t["pep_chain"], device=args.device)
            for allele, g in d.groupby("allele"):
                seq = alleles.set_index("allele").loc[allele, "hla_seq"]
                if args.scramble:
                    seq = "".join(rng.permutation(list(seq)))
                mean_lp, per_pos = conditional_scores(
                    model, tpl, seq, g["peptide"].tolist(), device,
                    n_orders=args.n_orders, seed=args.seed)
                out.append(pd.DataFrame({
                    "template": t["name"], "allele": allele, "peptide": g["peptide"].values,
                    "mpnn_mean_logp": mean_lp,
                    **{f"mpnn_logp_p{i+1}": per_pos[:, i] for i in range(9)}}))
                print(f"    {t['name']} {allele}: {len(g)} peptides", flush=True)
        tag = "_scrambled" if args.scramble else ""
        pd.concat(out).to_csv(os.path.join(args.output_dir, f"cond_scores{tag}.csv"), index=False)
        print(f"[*] wrote cond_scores{tag}.csv")


if __name__ == "__main__":
    main()
