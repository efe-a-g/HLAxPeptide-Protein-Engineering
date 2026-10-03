# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.2", "pyarrow", "numpy"]
# ///
"""Pool Boltz-2 per-residue embeddings over the HLA B and F binding pockets.

Full-chain mean pooling averages all 182 residues, which dilutes the peptide
binding groove: between-allele cosine similarity came out at 0.978 on average.
The B pocket holds the P2 anchor and the F pocket holds the C-terminal (POmega)
anchor, and those two anchors dominate complex stability, so pooling over them
separately should sharpen the allele signal.

Pocket definitions (Saper/Madden, via "The pockets guide to HLA class I
molecules", Biochem Soc Trans 2021):
  B: 7, 9, 24, 34, 45, 63, 66, 67, 70, 99                       (P2 anchor)
  F: 74, 77, 80, 81, 84, 95, 97, 114, 116, 123, 133, 143, 146, 147  (POmega)
The broader F set is used because it includes positions 97, 114 and 116, which
are among the most polymorphic residues across the 75 alleles in this dataset.

Residue numbering is 1-based over hla_seq, verified against hla_pseudoseq: the
34 NetMHCpan pseudosequence positions reproduce the stored pseudosequence
exactly for all 75/75 alleles.

Input is the per-residue arrays written by modal_boltz_embeddings.py
(embeddings/per_residue/<record_id>.npz, each holding s of shape (182, 384)).

Usage:
    uv run scripts/pocket_pool_embeddings.py
    uv run scripts/pocket_pool_embeddings.py --no-full-chain   # drop boltz_s_*
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POCKET_B = [7, 9, 24, 34, 45, 63, 66, 67, 70, 99]
POCKET_F = [74, 77, 80, 81, 84, 95, 97, 114, 116, 123, 133, 143, 146, 147]

# The 34 NetMHCpan pseudosequence positions, used to verify residue numbering.
PSEUDO_POS = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
              84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159,
              163, 167, 171]


def check_numbering(hla: pd.DataFrame) -> None:
    """Confirm hla_seq is 1-based standard HLA numbering before indexing into it."""
    bad = [
        r.allele
        for r in hla.itertuples()
        if "".join(r.hla_seq[p - 1] for p in PSEUDO_POS) != r.hla_pseudoseq
    ]
    if bad:
        raise SystemExit(
            f"Residue numbering check failed for {len(bad)} alleles (e.g. {bad[:3]}). "
            "hla_seq does not line up with hla_pseudoseq at the expected positions."
        )
    print(f"Numbering verified: pseudosequence positions match for all {len(hla)} alleles")


def cosine_spread(mat: np.ndarray) -> str:
    """Report pairwise cosine similarity across alleles, to show pooling sharpness."""
    unit = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sim = unit @ unit.T
    iu = np.triu_indices(len(mat), 1)
    return f"min {sim[iu].min():.4f}  mean {sim[iu].mean():.4f}  max {sim[iu].max():.4f}"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=root / "rasmussen_et_al_dataset.csv")
    p.add_argument("--per-residue", type=Path, default=root / "embeddings" / "per_residue")
    p.add_argument("--output", type=Path,
                   default=root / "embeddings" / "rasmussen_et_al_dataset_boltz_pockets.parquet")
    p.add_argument("--csv", action="store_true", help="Also write a .csv alongside the Parquet")
    p.add_argument("--no-full-chain", action="store_true",
                   help="Omit the full-chain mean (boltz_s_*) baseline columns")
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype={"thalf_hours": float})
    hla = df[["allele", "hla_seq", "hla_pseudoseq"]].drop_duplicates().reset_index(drop=True)
    check_numbering(hla)

    # Each .npz is keyed by the sanitised record id but stores the real allele.
    s_by_allele: dict[str, np.ndarray] = {}
    for path in sorted(args.per_residue.glob("*.npz")):
        with np.load(path) as f:
            s_by_allele[str(f["allele"])] = f["s"]
    missing = set(hla["allele"]) - set(s_by_allele)
    if missing:
        raise SystemExit(f"No per-residue embedding for {len(missing)} alleles: {sorted(missing)[:5]}")

    length = {s.shape[0] for s in s_by_allele.values()}
    if length != {len(hla["hla_seq"].iloc[0])}:
        raise SystemExit(f"Embedding length {length} does not match hla_seq length")

    blocks: dict[str, list[int] | None] = {"B": POCKET_B, "F": POCKET_F}
    if not args.no_full_chain:
        blocks["s"] = None  # None means pool over the whole chain

    frames = [pd.DataFrame({"allele": hla["allele"]})]
    for name, positions in blocks.items():
        pooled = np.stack([
            s_by_allele[a][[i - 1 for i in positions]].mean(axis=0) if positions
            else s_by_allele[a].mean(axis=0)
            for a in hla["allele"]
        ]).astype("float32")
        label = "full chain" if positions is None else f"pocket {name} ({len(positions)} residues)"
        print(f"{label:>32}: cosine across alleles  {cosine_spread(pooled)}")
        prefix = "boltz_s_" if positions is None else f"boltz_{name}_"
        frames.append(pd.DataFrame(pooled, columns=[f"{prefix}{i}" for i in range(pooled.shape[1])]))

    emb = pd.concat(frames, axis=1)
    out = df.merge(emb, on="allele", how="inner", validate="many_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {args.output} ({out.shape[0]} rows x {out.shape[1]} cols)")
    if args.csv:
        csv_path = args.output.with_suffix(".csv")
        out.to_csv(csv_path, index=False, float_format="%.6g")
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
