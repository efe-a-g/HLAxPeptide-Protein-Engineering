# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "boltz>=2.2",
#     "pandas>=2.2",
#     "pyarrow",
# ]
# ///
"""Augment the Rasmussen et al. dataset with Boltz-2 embeddings of hla_seq.

The dataset has ~28k rows but only one HLA sequence per allele (75 unique),
so Boltz is run once per allele and the embeddings are joined back onto rows.

Boltz-2's trunk produces a single (per-residue) representation `s` of shape
(L, 384) and a pair representation `z` of shape (L, L, 128). This script:

  1. writes one Boltz YAML per allele,
  2. runs `boltz predict --write_embeddings` over all of them in one call,
  3. collects `s` for each allele into <out>/hla_boltz_s.npz (per-residue),
  4. mean-pools `s` over residues and writes the dataset with one column per
     embedding dimension (boltz_s_0 ... boltz_s_383) to a Parquet file.

`z` stays in the Boltz output folder (embeddings_<id>.npz) since it is
~17 MB per allele.

Usage:
    uv run scripts/boltz_hla_embeddings.py                       # GPU, MSA server
    uv run scripts/boltz_hla_embeddings.py --accelerator cpu --limit 2   # quick test
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def allele_to_id(allele: str) -> str:
    """HLA-A*02:01 -> HLA-A_02_01 (safe for file names / Boltz record ids)."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", allele).strip("_")


def write_yamls(hla: pd.DataFrame, yaml_dir: Path, msa: str) -> None:
    yaml_dir.mkdir(parents=True, exist_ok=True)
    msa_line = "      msa: empty\n" if msa == "empty" else ""
    for row in hla.itertuples():
        (yaml_dir / f"{row.record_id}.yaml").write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {row.hla_seq}\n"
            f"{msa_line}"
        )


def run_boltz(yaml_dir: Path, boltz_out: Path, args: argparse.Namespace) -> None:
    cmd = [
        "boltz", "predict", str(yaml_dir),
        "--out_dir", str(boltz_out),
        "--write_embeddings",
        "--accelerator", args.accelerator,
        "--recycling_steps", str(args.recycling_steps),
        # Embeddings come from the trunk, before diffusion, so the structure
        # sampling budget does not affect them. Keep it small to save time.
        "--sampling_steps", str(args.sampling_steps),
        "--diffusion_samples", "1",
        "--output_format", "pdb",
    ]
    if args.msa == "server":
        cmd.append("--use_msa_server")
    if args.cache:
        cmd += ["--cache", str(args.cache)]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_s(pred_dir: Path, record_id: str) -> np.ndarray:
    path = pred_dir / record_id / f"embeddings_{record_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Boltz embeddings for {record_id}: {path}")
    s = np.load(path)["s"]
    return s.reshape(s.shape[-2], s.shape[-1])  # drop batch dim -> (L, D)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=root / "rasmussen_et_al_dataset.csv")
    p.add_argument("--out-dir", type=Path, default=root / "embeddings")
    p.add_argument("--output", type=Path, default=None,
                   help="Augmented dataset path (default: <out-dir>/rasmussen_et_al_dataset_boltz.parquet)")
    p.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--msa", choices=["server", "empty"], default="server",
                   help="'server' sends the HLA sequences to the ColabFold MSA server; "
                        "'empty' runs single-sequence mode (offline, lower quality)")
    p.add_argument("--recycling-steps", type=int, default=3)
    p.add_argument("--sampling-steps", type=int, default=10)
    p.add_argument("--cache", type=Path, default=None, help="Boltz model cache dir (default ~/.boltz)")
    p.add_argument("--limit", type=int, default=None, help="Only embed the first N alleles (for testing)")
    p.add_argument("--skip-boltz", action="store_true",
                   help="Don't run Boltz; just collect existing embeddings from --out-dir")
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype={"thalf_hours": float})
    hla = df[["allele", "hla_seq"]].drop_duplicates().reset_index(drop=True)
    if hla["allele"].duplicated().any():
        sys.exit("Some alleles map to more than one hla_seq; expected one sequence per allele.")
    hla["record_id"] = hla["allele"].map(allele_to_id)
    if args.limit:
        hla = hla.head(args.limit)
    print(f"{len(df)} rows, embedding {len(hla)} unique HLA sequences")

    yaml_dir = args.out_dir / "boltz_inputs"
    boltz_out = args.out_dir / "boltz"
    pred_dir = boltz_out / f"boltz_results_{yaml_dir.name}" / "predictions"

    if not args.skip_boltz:
        write_yamls(hla, yaml_dir, args.msa)
        # Boltz skips records that already have predictions, so re-running resumes.
        run_boltz(yaml_dir, boltz_out, args)

    per_residue = {r.allele: load_s(pred_dir, r.record_id) for r in hla.itertuples()}
    np.savez_compressed(args.out_dir / "hla_boltz_s.npz", **per_residue)

    pooled = np.stack([per_residue[a].mean(axis=0) for a in hla["allele"]])
    emb = pd.DataFrame(pooled, columns=[f"boltz_s_{i}" for i in range(pooled.shape[1])])
    emb.insert(0, "allele", hla["allele"].values)

    out = df.merge(emb, on="allele", how="inner", validate="many_to_one")
    output = args.output or args.out_dir / "rasmussen_et_al_dataset_boltz.parquet"
    out.to_parquet(output, index=False)
    print(f"Wrote {output} ({out.shape[0]} rows x {out.shape[1]} cols)")
    print(f"Wrote {args.out_dir / 'hla_boltz_s.npz'} (per-residue s, keyed by allele)")


if __name__ == "__main__":
    main()
