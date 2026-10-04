"""Co-fold pMHC-I as the native TRIMER and extract the same confidence features.

Control for the alpha1/alpha2-only run: chain A = full alpha ectodomain
(alpha1+alpha2+alpha3, 274 aa), chain B = beta-2-microglobulin (99 aa),
chain C = peptide (9 aa). 382 tokens vs 191 for the two-chain run.

The question is whether the weak signal in the 191-token run (best per-allele
SCC 0.148) was limited by the missing alpha3/b2m scaffold -- Boltz reported low
absolute confidence there (hla_plddt ~0.57), consistent with an unstable
truncated fold rather than with a genuine absence of peptide-specific signal.

Feature extraction is identical to fold_complexes.py so the two runs are
directly comparable. Pocket indices stay 1-based over alpha1/alpha2, which
occupies the first 182 residues of chain A, so the same numbering applies.

Differences from fold_complexes.py, both deliberate:
  - No parquet write. pyarrow is absent from the Boltz image and to_parquet()
    was what killed the previous run AFTER all folding had completed. jsonl and
    npz only; aggregate locally.
  - Embeddings are written per chunk rather than once at the end, so a late
    failure cannot discard them.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POCKET_B = [7, 9, 24, 34, 45, 63, 66, 67, 70, 99]
POCKET_F = [74, 77, 80, 81, 84, 95, 97, 114, 116, 123, 133, 143, 146, 147]
PSEUDO_POS = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
              84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159,
              163, 167, 171]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "_", s).strip("_")


def check_numbering(df: pd.DataFrame) -> None:
    """chain_a[:182] must reproduce the stored pseudosequence before indexing pockets."""
    sub = df[["allele", "chain_a", "hla_pseudoseq"]].drop_duplicates("allele")
    bad = [r.allele for r in sub.itertuples()
           if "".join(r.chain_a[p - 1] for p in PSEUDO_POS) != r.hla_pseudoseq]
    if bad:
        sys.exit(f"Numbering check failed for {len(bad)} alleles, e.g. {bad[:3]}")
    print(f"[ok] pocket numbering verified against pseudosequence for "
          f"all {len(sub)} alleles", flush=True)


def write_yaml(path: Path, chains: list[tuple[str, str, bool]]) -> None:
    """chains = [(chain_id, sequence, needs_msa), ...]"""
    lines = ["version: 1", "sequences:"]
    for cid, seq, needs_msa in chains:
        lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
        if not needs_msa:
            lines.append("      msa: empty")
    path.write_text("\n".join(lines) + "\n")


def run_boltz(yaml_dir: Path, out_dir: Path, args) -> Path:
    cmd = ["boltz", "predict", str(yaml_dir), "--out_dir", str(out_dir),
           "--cache", "/root/.boltz", "--accelerator", "gpu",
           "--recycling_steps", str(args.recycling_steps),
           "--sampling_steps", str(args.sampling_steps),
           "--diffusion_samples", "1", "--output_format", "pdb",
           "--use_msa_server"]
    if not args.no_embeddings:
        cmd.append("--write_embeddings")
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / f"boltz_results_{yaml_dir.name}"


def extract(pred_dir: Path, rid: str, n_a: int, n_b2m: int, n_pep: int):
    """Identical feature set to fold_complexes.py; peptide offset accounts for b2m."""
    feat: dict[str, float] = {}
    off = n_a + n_b2m
    pep = list(range(off, off + n_pep))
    bidx = [p - 1 for p in POCKET_B]
    findx = [p - 1 for p in POCKET_F]

    cj = pred_dir / f"confidence_{rid}_model_0.json"
    if cj.exists():
        c = json.loads(cj.read_text())
        for k, v in c.items():
            if isinstance(v, (int, float)):
                feat[f"conf_{k}"] = float(v)
        pci = c.get("pair_chains_iptm")
        if isinstance(pci, dict):
            for a, row in pci.items():
                if isinstance(row, dict):
                    for b, v in row.items():
                        if a != b and isinstance(v, (int, float)):
                            feat[f"iptm_pair_{a}{b}"] = float(v)
        cp = c.get("chains_ptm")
        if isinstance(cp, dict):
            for k, v in cp.items():
                if isinstance(v, (int, float)):
                    feat[f"ptm_chain_{k}"] = float(v)

    pl = pred_dir / f"plddt_{rid}_model_0.npz"
    if pl.exists():
        with np.load(pl) as z:
            a = z[z.files[0]].squeeze()
        if a.ndim == 1 and a.shape[0] >= off + n_pep:
            pp, hp = a[pep], a[:n_a]
            feat["pep_plddt_mean"] = float(pp.mean())
            feat["pep_plddt_min"] = float(pp.min())
            feat["pep_plddt_std"] = float(pp.std())
            for i, v in enumerate(pp, start=1):
                feat[f"pep_plddt_P{i}"] = float(v)
            feat["hla_plddt_mean"] = float(hp.mean())
            feat["pocketB_plddt_mean"] = float(hp[bidx].mean())
            feat["pocketF_plddt_mean"] = float(hp[findx].mean())
            if n_b2m:
                feat["b2m_plddt_mean"] = float(a[n_a:n_a + n_b2m].mean())

    pae = pred_dir / f"pae_{rid}_model_0.npz"
    if pae.exists():
        with np.load(pae) as z:
            m = z[z.files[0]].squeeze()
        if m.ndim == 2 and m.shape[0] >= off + n_pep:
            feat["pae_pep_pep"] = float(m[np.ix_(pep, pep)].mean())
            feat["pae_pep_to_B"] = float(m[np.ix_(pep, bidx)].mean())
            feat["pae_pep_to_F"] = float(m[np.ix_(pep, findx)].mean())
            feat["pae_B_to_pep"] = float(m[np.ix_(bidx, pep)].mean())
            feat["pae_F_to_pep"] = float(m[np.ix_(findx, pep)].mean())
            feat["pae_pep_to_hla"] = float(m[np.ix_(pep, range(n_a))].mean())

    emb: dict[str, np.ndarray] = {}
    ej = pred_dir / f"embeddings_{rid}.npz"
    if ej.exists():
        with np.load(ej) as z:
            s = z["s"]
        s = s.reshape(s.shape[-2], s.shape[-1])
        if s.shape[0] >= off + n_pep:
            emb["pep"] = s[pep].astype("float32")
            emb["pocketB"] = s[bidx].mean(axis=0).astype("float32")
            emb["pocketF"] = s[findx].mean(axis=0).astype("float32")
    return feat, emb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="trimer_sample.csv")
    p.add_argument("--out", default="out")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=10)
    p.add_argument("--keep_structures", type=int, default=8)
    p.add_argument("--no_b2m", action="store_true", help="omit b2m (alpha1+2+3 + peptide only)")
    p.add_argument("--no_embeddings", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work"); work.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    check_numbering(df)
    n_b2m = 0 if args.no_b2m else len(df["chain_b2m"].iloc[0])
    print(f"[in] {len(df)} complexes | chain_a {len(df.chain_a.iloc[0])} aa | "
          f"b2m {n_b2m} aa | peptide 9 aa | "
          f"{len(df.chain_a.iloc[0]) + n_b2m + 9} tokens", flush=True)

    jsonl = out / "features.jsonl"
    kept = 0
    probed = False

    for ci in range(0, len(df), args.chunk):
        chunk = df.iloc[ci:ci + args.chunk]
        ydir = work / f"chunk_{ci:05d}"
        if ydir.exists():
            shutil.rmtree(ydir)
        ydir.mkdir(parents=True)
        meta = {}
        for r in chunk.itertuples():
            rid = f"{sanitize(r.allele)}__{r.peptide}"
            meta[rid] = r
            chains = [("A", r.chain_a, True)]
            if n_b2m:
                chains.append(("B", r.chain_b2m, True))
            chains.append(("C" if n_b2m else "B", r.peptide, False))
            write_yaml(ydir / f"{rid}.yaml", chains)

        res = run_boltz(ydir, work / f"out_{ci:05d}", args)
        preds = res / "predictions"
        emb_chunk: dict[str, np.ndarray] = {}

        with jsonl.open("a") as fh:
            for rid, r in meta.items():
                pdir = preds / rid
                if not pdir.exists():
                    print(f"[miss] {rid}", flush=True)
                    continue
                if not probed:
                    for f in sorted(pdir.iterdir()):
                        extra = ""
                        if f.suffix == ".npz":
                            with np.load(f) as z:
                                extra = " keys=" + ",".join(f"{k}{z[k].shape}" for k in z.files)
                        print(f"[probe] {f.name} ({f.stat().st_size} B){extra}", flush=True)
                    probed = True
                feat, emb = extract(pdir, rid, len(r.chain_a), n_b2m, len(r.peptide))
                fh.write(json.dumps({"allele": r.allele, "peptide": r.peptide,
                                     "thalf_hours": float(r.thalf_hours),
                                     "record_id": rid, **feat}) + "\n")
                for k, v in emb.items():
                    emb_chunk[f"{rid}__{k}"] = v
                if kept < args.keep_structures:
                    for pdb in list(pdir.glob("*_model_0.pdb")):
                        shutil.copyfile(pdb, out / "structures" / f"{rid}.pdb")
                        kept += 1
                        break

        # write embeddings per chunk so a late failure cannot discard them
        if emb_chunk:
            np.savez_compressed(out / f"embeddings_{ci:05d}.npz", **emb_chunk)
        shutil.rmtree(work / f"out_{ci:05d}", ignore_errors=True)
        shutil.rmtree(ydir, ignore_errors=True)
        n = sum(1 for _ in jsonl.open())
        print(f"[progress] {n}/{len(df)} complexes extracted", flush=True)

    print(f"[done] {sum(1 for _ in jsonl.open())} rows -> {jsonl}", flush=True)


if __name__ == "__main__":
    main()
