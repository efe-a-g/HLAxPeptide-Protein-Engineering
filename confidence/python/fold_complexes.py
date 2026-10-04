"""Co-fold peptide-HLA class I complexes with Boltz-2 and extract per-complex features.

Runs inside the Modal job container. Two passes:

  1. MSA pass  -- one monomer YAML per UNIQUE hla_seq, --use_msa_server, then the
     fetched .a3m files are located and reused. Only a handful of unique HLA
     sequences exist (75 in the full dataset), so this is done once instead of
     once per complex. The 9-mer peptide chain gets `msa: empty`: a 9-residue
     query has no useful homologs and the fetch would dominate runtime.

  2. Fold pass -- chain A = hla_seq (alpha1/alpha2, 182 aa), chain B = peptide (9 aa).
     Beta-2-microglobulin is deliberately omitted: hla_seq covers alpha1+alpha2 only,
     and b2m's native interface is mostly with alpha3, so including it would present
     b2m a truncated binding site. 191 tokens instead of 290.

Features come from the confidence head (pLDDT / PAE / ipTM), NOT from the affinity
head -- Boltz-2's affinity module requires a `ligand` binder chain and cannot score
a peptide. Trunk embeddings are written in the same pass since the co-fold is the
expensive part.

Results are appended to out/features.jsonl after every chunk, so a deadline harvests
partial results rather than losing the run.
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

# Pocket definitions, 1-based over hla_seq (Saper/Madden numbering), matching
# scripts/pocket_pool_embeddings.py on the dev branch so features stay comparable.
POCKET_B = [7, 9, 24, 34, 45, 63, 66, 67, 70, 99]
POCKET_F = [74, 77, 80, 81, 84, 95, 97, 114, 116, 123, 133, 143, 146, 147]
PSEUDO_POS = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
              84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159,
              163, 167, 171]


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "_", s).strip("_")


def check_numbering(df: pd.DataFrame) -> None:
    """hla_seq must be 1-based standard numbering before we index pockets into it."""
    hla = df[["allele", "hla_seq", "hla_pseudoseq"]].drop_duplicates()
    bad = [r.allele for r in hla.itertuples()
           if "".join(r.hla_seq[p - 1] for p in PSEUDO_POS) != r.hla_pseudoseq]
    if bad:
        sys.exit(f"Residue numbering check failed for {len(bad)} alleles, e.g. {bad[:3]}")
    print(f"[ok] pseudosequence positions verified for all {len(hla)} alleles", flush=True)


def run_boltz(yaml_dir: Path, out_dir: Path, args, use_msa_server: bool,
              write_embeddings: bool) -> Path:
    cmd = [
        "boltz", "predict", str(yaml_dir),
        "--out_dir", str(out_dir),
        "--cache", "/root/.boltz",
        "--accelerator", "gpu",
        "--recycling_steps", str(args.recycling_steps),
        "--sampling_steps", str(args.sampling_steps),
        "--diffusion_samples", "1",
        "--output_format", "pdb",
    ]
    if write_embeddings:
        cmd.append("--write_embeddings")
    if use_msa_server:
        cmd.append("--use_msa_server")
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / f"boltz_results_{yaml_dir.name}"


def msa_pass(df: pd.DataFrame, work: Path, args) -> dict[str, Path]:
    """Fetch one MSA per unique hla_seq. Returns {hla_seq: a3m path}."""
    hla = df[["allele", "hla_seq"]].drop_duplicates("hla_seq")
    ydir = work / "msa_inputs"
    ydir.mkdir(parents=True, exist_ok=True)
    rid_to_seq = {}
    for r in hla.itertuples():
        rid = sanitize(r.allele)
        rid_to_seq[rid] = r.hla_seq
        (ydir / f"{rid}.yaml").write_text(
            "version: 1\nsequences:\n  - protein:\n      id: A\n"
            f"      sequence: {r.hla_seq}\n"
        )
    print(f"[msa] fetching MSAs for {len(rid_to_seq)} unique HLA sequences", flush=True)
    res = run_boltz(ydir, work / "msa_out", args, use_msa_server=True,
                    write_embeddings=False)

    found = sorted(res.rglob("*.a3m"))
    print(f"[msa] located {len(found)} a3m files under {res}", flush=True)
    for p in found[:4]:
        print("   ", p.relative_to(res), p.stat().st_size, "bytes", flush=True)

    seq_to_a3m: dict[str, Path] = {}
    store = work / "msa_cache"
    store.mkdir(parents=True, exist_ok=True)
    for rid, seq in rid_to_seq.items():
        hits = [p for p in found if p.stem.startswith(rid)]
        if hits:
            dest = store / f"{rid}.a3m"
            shutil.copyfile(hits[0], dest)
            seq_to_a3m[seq] = dest
    print(f"[msa] cached {len(seq_to_a3m)}/{len(rid_to_seq)} MSAs", flush=True)
    return seq_to_a3m


def write_complex_yaml(path: Path, hla_seq: str, peptide: str, a3m: Path | None) -> None:
    msa_line = f"      msa: {a3m}\n" if a3m else ""
    path.write_text(
        "version: 1\nsequences:\n"
        "  - protein:\n      id: A\n"
        f"      sequence: {hla_seq}\n{msa_line}"
        "  - protein:\n      id: B\n"
        f"      sequence: {peptide}\n"
        "      msa: empty\n"
    )


def describe_outputs(pred_dir: Path) -> None:
    """One-off format discovery so a wrong assumption is visible in the log."""
    print(f"[probe] contents of {pred_dir.name}:", flush=True)
    for p in sorted(pred_dir.iterdir()):
        line = f"    {p.name}  ({p.stat().st_size} B)"
        if p.suffix == ".npz":
            with np.load(p) as z:
                line += "  keys=" + ",".join(f"{k}{z[k].shape}" for k in z.files)
        elif p.suffix == ".json":
            try:
                line += "  keys=" + ",".join(json.loads(p.read_text()).keys())
            except Exception:
                pass
        print(line, flush=True)


def extract(pred_dir: Path, rid: str, n_hla: int, n_pep: int) -> tuple[dict, dict]:
    """Scalar features from the confidence head, plus trunk embedding slices."""
    feat: dict[str, float] = {}
    pep = list(range(n_hla, n_hla + n_pep))
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
        if a.ndim == 1 and a.shape[0] >= n_hla + n_pep:
            pp, hp = a[pep], a[:n_hla]
            feat["pep_plddt_mean"] = float(pp.mean())
            feat["pep_plddt_min"] = float(pp.min())
            feat["pep_plddt_std"] = float(pp.std())
            for i, v in enumerate(pp, start=1):
                feat[f"pep_plddt_P{i}"] = float(v)
            feat["hla_plddt_mean"] = float(hp.mean())
            feat["pocketB_plddt_mean"] = float(hp[bidx].mean())
            feat["pocketF_plddt_mean"] = float(hp[findx].mean())

    pae = pred_dir / f"pae_{rid}_model_0.npz"
    if pae.exists():
        with np.load(pae) as z:
            m = z[z.files[0]].squeeze()
        if m.ndim == 2 and m.shape[0] >= n_hla + n_pep:
            feat["pae_pep_pep"] = float(m[np.ix_(pep, pep)].mean())
            feat["pae_pep_to_B"] = float(m[np.ix_(pep, bidx)].mean())
            feat["pae_pep_to_F"] = float(m[np.ix_(pep, findx)].mean())
            feat["pae_B_to_pep"] = float(m[np.ix_(bidx, pep)].mean())
            feat["pae_F_to_pep"] = float(m[np.ix_(findx, pep)].mean())
            feat["pae_pep_to_hla"] = float(m[np.ix_(pep, range(n_hla))].mean())

    emb: dict[str, np.ndarray] = {}
    ej = pred_dir / f"embeddings_{rid}.npz"
    if ej.exists():
        with np.load(ej) as z:
            s = z["s"]
        s = s.reshape(s.shape[-2], s.shape[-1])
        if s.shape[0] >= n_hla + n_pep:
            emb["pep"] = s[pep].astype("float32")
            emb["pocketB"] = s[bidx].mean(axis=0).astype("float32")
            emb["pocketF"] = s[findx].mean(axis=0).astype("float32")
    return feat, emb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="sample.csv")
    p.add_argument("--out", default="out")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--keep_structures", type=int, default=12)
    p.add_argument("--no_embeddings", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    check_numbering(df)
    print(f"[in] {len(df)} complexes, {df.allele.nunique()} alleles, "
          f"{df.peptide.nunique()} peptides", flush=True)

    seq_to_a3m = msa_pass(df, work, args)

    jsonl = out / "features.jsonl"
    emb_store: dict[str, np.ndarray] = {}
    n_done = 0
    probed = False
    kept = 0

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
            write_complex_yaml(ydir / f"{rid}.yaml", r.hla_seq, r.peptide,
                               seq_to_a3m.get(r.hla_seq))
        need_server = any(r.hla_seq not in seq_to_a3m for r in chunk.itertuples())
        if need_server:
            print("[warn] some MSAs missing from cache; falling back to msa server "
                  "for this chunk", flush=True)

        res = run_boltz(ydir, work / f"out_{ci:05d}", args,
                        use_msa_server=need_server,
                        write_embeddings=not args.no_embeddings)
        preds = res / "predictions"

        with jsonl.open("a") as fh:
            for rid, r in meta.items():
                pdir = preds / rid
                if not pdir.exists():
                    print(f"[miss] no prediction dir for {rid}", flush=True)
                    continue
                if not probed:
                    describe_outputs(pdir)
                    probed = True
                feat, emb = extract(pdir, rid, len(r.hla_seq), len(r.peptide))
                row = {"allele": r.allele, "peptide": r.peptide,
                       "thalf_hours": float(r.thalf_hours), "record_id": rid, **feat}
                fh.write(json.dumps(row) + "\n")
                for k, v in emb.items():
                    emb_store[f"{rid}__{k}"] = v
                if kept < args.keep_structures:
                    for cif in list(pdir.glob("*_model_0.pdb")):
                        shutil.copyfile(cif, out / "structures" / f"{rid}.pdb")
                        kept += 1
                        break
                n_done += 1

        shutil.rmtree(work / f"out_{ci:05d}", ignore_errors=True)
        shutil.rmtree(ydir, ignore_errors=True)
        print(f"[progress] {n_done}/{len(df)} complexes extracted", flush=True)

    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    feats = pd.DataFrame(rows)
    feats.to_parquet(out / "features.parquet", index=False)
    if emb_store:
        np.savez_compressed(out / "embeddings.npz", **emb_store)
    print(f"[done] {len(feats)} rows x {feats.shape[1]} cols -> {out/'features.parquet'}",
          flush=True)
    print("[cols]", ", ".join(feats.columns), flush=True)


if __name__ == "__main__":
    main()
