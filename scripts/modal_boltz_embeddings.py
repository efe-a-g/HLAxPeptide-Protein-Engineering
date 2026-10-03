"""Compute Boltz-2 embeddings of the HLA sequences on Modal GPUs.

The Rasmussen dataset has ~28k rows but only one HLA sequence per allele
(75 unique), so Boltz runs once per allele and the result is joined back
onto every row locally.

Boltz-2's trunk produces a single representation `s` of shape (L, 384) and a
pair representation `z` of shape (L, L, 128). Alleles are split into shards
that run in parallel containers; each shard writes its per-residue `s` (and
the Boltz structure outputs) to a Modal Volume and returns the mean-pooled
vector, which is small enough to send back over the wire.

Run:
    modal run scripts/modal_boltz_embeddings.py --limit 2     # smoke test
    modal run scripts/modal_boltz_embeddings.py               # all 75 alleles

Then fetch the per-residue embeddings (optional, ~21 MB):
    modal volume get boltz-hla-out <run>/per_residue embeddings/per_residue

Set BOLTZ_GPU to change the GPU type (default A10), e.g. BOLTZ_GPU=L40S.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import modal

GPU = os.environ.get("BOLTZ_GPU", "A10")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # boltz[cuda] pulls in the cuequivariance triangle kernels. Boltz imports
    # them unconditionally unless --no_kernels is passed, so they must be here.
    .pip_install("boltz[cuda]==2.2.1", "pyarrow")
)

# Model weights (~3 GB) are downloaded on first run and reused afterwards.
cache_vol = modal.Volume.from_name("boltz-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("boltz-hla-out", create_if_missing=True)

app = modal.App("boltz-hla-embeddings", image=image)


def allele_to_id(allele: str) -> str:
    """HLA-A*02:01 -> HLA-A_02_01 (safe for file names / Boltz record ids)."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", allele).strip("_")


@app.function(
    gpu=GPU,
    volumes={"/cache": cache_vol, "/out": out_vol},
    timeout=4 * 60 * 60,
)
def embed_shard(
    shard: list[tuple[str, str]],
    run: str,
    msa: str = "server",
    recycling_steps: int = 3,
    sampling_steps: int = 10,
    use_kernels: bool = True,
) -> dict[str, list[float]]:
    """Embed one shard of (allele, hla_seq) pairs. Returns mean-pooled vectors."""
    import numpy as np

    yaml_dir = Path("/tmp/boltz_inputs")
    yaml_dir.mkdir(parents=True, exist_ok=True)
    msa_line = "      msa: empty\n" if msa == "empty" else ""
    for allele, seq in shard:
        (yaml_dir / f"{allele_to_id(allele)}.yaml").write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {seq}\n"
            f"{msa_line}"
        )

    boltz_out = Path("/out") / run / "boltz"
    cmd = [
        "boltz", "predict", str(yaml_dir),
        "--out_dir", str(boltz_out),
        "--write_embeddings",
        "--cache", "/cache",
        "--accelerator", "gpu",
        "--recycling_steps", str(recycling_steps),
        # Embeddings come from the trunk, before diffusion, so the structure
        # sampling budget does not affect them. Keep it small to save time.
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", "1",
        "--output_format", "pdb",
    ]
    if msa == "server":
        cmd.append("--use_msa_server")
    if not use_kernels:
        cmd.append("--no_kernels")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    pred_dir = boltz_out / f"boltz_results_{yaml_dir.name}" / "predictions"
    per_residue_dir = Path("/out") / run / "per_residue"
    per_residue_dir.mkdir(parents=True, exist_ok=True)

    pooled: dict[str, list[float]] = {}
    for allele, _ in shard:
        rid = allele_to_id(allele)
        path = pred_dir / rid / f"embeddings_{rid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Boltz wrote no embeddings for {allele}: {path}")
        s = np.load(path)["s"]
        s = s.reshape(s.shape[-2], s.shape[-1])  # drop batch dim -> (L, D)
        np.savez_compressed(per_residue_dir / f"{rid}.npz", s=s, allele=allele)
        pooled[allele] = s.mean(axis=0).astype("float32").tolist()
        print(f"{allele}: s {s.shape}", flush=True)

    out_vol.commit()
    return pooled


@app.function(volumes={"/out": out_vol}, timeout=30 * 60)
def build_dataset(csv_bytes: bytes, pooled: dict[str, list[float]], run: str, name: str) -> str:
    """Join the pooled embeddings onto every row and write Parquet to the volume."""
    import io

    import pandas as pd

    df = pd.read_csv(io.BytesIO(csv_bytes), dtype={"thalf_hours": float})
    alleles = list(pooled)
    dim = len(pooled[alleles[0]])
    emb = pd.DataFrame(
        [pooled[a] for a in alleles],
        columns=[f"boltz_s_{i}" for i in range(dim)],
    )
    emb.insert(0, "allele", alleles)

    out = df.merge(emb, on="allele", how="inner", validate="many_to_one")
    path = Path("/out") / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    out_vol.commit()
    print(f"{out.shape[0]} rows x {out.shape[1]} cols -> {path}")
    return f"{out.shape[0]} rows x {out.shape[1]} cols"


def read_unique_hla(path: Path) -> list[tuple[str, str]]:
    """Unique (allele, hla_seq) pairs, using only the stdlib so the local side
    needs nothing but `modal` installed."""
    import csv

    seen: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            allele, seq = row["allele"], row["hla_seq"]
            if seen.setdefault(allele, seq) != seq:
                sys.exit(f"{allele} maps to more than one hla_seq; expected one per allele.")
    return list(seen.items())


@app.local_entrypoint()
def main(
    input: str = "rasmussen_et_al_dataset.csv",
    output: str = "rasmussen_et_al_dataset_boltz.parquet",
    run: str = "v1",
    msa: str = "server",
    shards: int = 4,
    limit: int = 0,
    recycling_steps: int = 3,
    sampling_steps: int = 10,
    no_kernels: bool = False,
):
    root = Path(__file__).resolve().parent.parent
    in_path = root / input if not Path(input).is_absolute() else Path(input)

    pairs = read_unique_hla(in_path)
    if limit:
        pairs = pairs[:limit]
    print(f"Embedding {len(pairs)} unique HLA sequences on {GPU}")

    n = max(1, min(shards, len(pairs)))
    batches = [pairs[i::n] for i in range(n)]
    pooled: dict[str, list[float]] = {}
    for result in embed_shard.starmap(
        [(b, run, msa, recycling_steps, sampling_steps, not no_kernels) for b in batches]
    ):
        pooled.update(result)

    missing = {a for a, _ in pairs} - set(pooled)
    if missing:
        sys.exit(f"No embeddings returned for {len(missing)} alleles: {sorted(missing)[:5]}")

    shape = build_dataset.remote(in_path.read_bytes(), pooled, run, output)
    print(
        f"Wrote {shape} to the volume. Fetch the results with:\n"
        f"  modal volume get boltz-hla-out {run}/{output} embeddings/{output}\n"
        f"  modal volume get boltz-hla-out {run}/per_residue embeddings/per_residue"
    )
