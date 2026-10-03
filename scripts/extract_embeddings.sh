#!/usr/bin/env bash
# Extract the Boltz-2 embedding parquets from origin/dev into embeddings/.
#
# MUST be run through a byte-safe redirect (bash, not PowerShell). A PowerShell
# text redirect corrupts these files: it prepends a UTF-8 BOM, appends CRLF, and
# replaces every non-UTF-8-decodable byte with U+FFFD (ef bf bd). That loss is
# irreversible -- the first copies handed to this worktree were damaged that way
# (41 MB / 14 MB instead of 24 MB / 8 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p embeddings
git cat-file blob origin/dev:embeddings/rasmussen_et_al_dataset_boltz_pockets.parquet \
  > embeddings/boltz_pockets.parquet
git cat-file blob origin/dev:embeddings/rasmussen_et_al_dataset_boltz.parquet \
  > embeddings/boltz_full.parquet
for f in embeddings/*.parquet; do
  [ "$(head -c 4 "$f")" = "PAR1" ] && [ "$(tail -c 4 "$f")" = "PAR1" ] \
    || { echo "CORRUPT: $f" >&2; exit 1; }
  echo "ok: $f ($(wc -c < "$f") bytes)"
done
