# Research log — peptide-HLA stability with protein foundation models

**Question.** Can open protein foundation models improve prediction of peptide-HLA class I
complex stability (half-life) over a sequence-only supervised baseline?

**Stance.** The organisers asked *"are they useful for this problem"*, not *"prove they are
useful"*. A well-supported negative result is a valid answer and is reported as such.

Dataset: Rasmussen et al. (2016), *J Immunol* 197(4):1517–1524 — 28,166 peptide-HLA
half-life measurements.

Machine: Intel Core Ultra 7 258V, 8 cores, 31.5 GB RAM. **No CUDA** (Intel Arc 140V iGPU
only) — all training is CPU, all structure prediction is treated as already-done.

---

## Entry 0 — Environment and data verification

`uv sync` on Python 3.12 → torch 2.14.1 (CPU), pandas 3.0.6, numpy 2.5.3, sklearn 1.9.1.
Added `pyarrow` (parquet engine, not in the original pin) and `transformers` (for ESM-2).

### Dataset facts (verified, `python/verify_data.py`)

| property | value |
|---|---|
| rows | 28,166 |
| unique alleles | 75 |
| unique peptides | 5,633 |
| peptide length | 9 only |
| `hla_seq` length | 182 |
| `hla_pseudoseq` length | 34 |
| `thalf_hours` | min 0.0, median 1.10, max 256.7, no NaN |
| duplicated (allele, peptide) | 0 |
| non-standard amino acids | 0 |

Two facts worth flagging that were not in the brief:

1. **5,679 rows (20.2%) have `thalf_hours` == 0.0** — exactly zero, not small. Under the
   NetMHC transform `s = 2^(-t0/thalf)` these all map to `s = 0`. So a fifth of the target
   distribution is a single point mass at the boundary. This caps achievable correlation and
   makes RMSE-in-hours a poor headline metric.
2. **74 unique pseudosequences for 75 alleles** — one pseudosequence is shared by two
   alleles. The three engineered constructs (`HLA-B*14:01(C67S)`, `HLA-B*14:02(C67S)`,
   `HLA-B*39:06(C67S)`) are present as separate alleles. C67 is not among the 34 NetMHCpan
   contact residues, so a C67S mutant is **invisible** to the pseudosequence encoding: the
   baseline cannot distinguish such a construct from its wild type.

### Data incident — the supplied embeddings were corrupted

`embeddings/boltz_pockets.parquet` and `boltz_full.parquet` as handed over **would not load**:
`ArrowInvalid: Parquet magic bytes not found in footer`.

Diagnosis from a hex dump: a UTF-8 BOM (`ef bb bf`) prepended, a trailing `CRLF` (`0d 0a`)
appended, and interior binary bytes replaced by `ef bf bd` — the UTF-8 encoding of U+FFFD
REPLACEMENT CHARACTER. This is the signature of extracting a binary blob through a
PowerShell **text** redirect: the bytes were decoded as text, every non-decodable byte
collapsed to U+FFFD, then re-encoded as UTF-8. The collapse is many-to-one, so it is
**irreversible** — the damaged files could not be repaired, only replaced.

Confirmed by size: the git blobs are 24,172,035 and 8,287,636 bytes; the damaged copies were
41,118,458 and 13,999,488 — inflated ~1.7x, consistent with multi-byte U+FFFD substitution.

Fixed by re-extracting with `git cat-file blob` through a byte-safe bash redirect:
`scripts/extract_embeddings.sh`, which verifies `PAR1` at head and tail and refuses to
proceed otherwise. **Lesson recorded: never move binary files through a PowerShell redirect.**

### Embedding facts (after recovery)

| | `boltz_pockets.parquet` | `boltz_full.parquet` |
|---|---|---|
| rows | 28,166 | 28,166 |
| blocks | `boltz_B_*` 384, `boltz_F_*` 384, `boltz_s_*` 384 | `boltz_s_*` 384 |
| **distinct embedding rows** | **75** | **75** |
| row order vs CSV | identical | identical |
| value range | [-1008.995, 276.498] | [-860.738, 179.618] |
| mean L2 norm | 1655.9 | 860.5 |

Two consequences that shape the whole experiment:

- **The Boltz-2 embeddings are per-allele: 75 distinct vectors for 28,166 rows.** They carry
  *zero* peptide information. They can only replace or augment the HLA side. Any comparison
  that feeds "Boltz-2 vs BLOSUM" without giving the Boltz model its own peptide
  representation is measuring nothing but the peptide encoder.
- **Scale mismatch is severe.** Boltz features reach magnitude ~1000 with mean L2 norm 1656,
  against BLOSUM50/5 inputs in roughly [-1, 3]. Fed raw into a 60-unit *sigmoid* MLP, the
  first layer saturates immediately and no gradient flows. Standardization (z-score, fit on
  train only) is therefore mandatory, not cosmetic — this alone would be enough to produce a
  spurious "foundation models don't work" result if skipped.

---

## Entry 1 — Baseline reproduction

`python/train_baseline.py`, unmodified. BLOSUM50 peptide(9) + pseudoseq(34) = 43x20 = 860
dims → 60-unit sigmoid MLP → sigmoid output. Target `s = 2^(-t0/thalf)`, t0 = 1 h.
25 epochs, Adam lr 1e-3, batch 128, weight decay 1e-5, best-validation checkpoint.

**Timing: 13–15 s per run** (38 s for the first, cold-start). Full dataset, all three splits.
Well under the 10-minute budget, so **no subsampling anywhere** — every number below uses all
28,166 rows.

### Random split, seed 42 — vs committed reference

| metric | reference | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| mean_allele_pcc | 0.5158 | 0.5300 | 0.5259 | 0.5323 |
| global_pcc_score | 0.7151 | 0.7242 | 0.7190 | 0.7195 |
| global_scc_score | 0.7239 | 0.7330 | 0.7259 | 0.7263 |
| auc_1h | 0.8686 | 0.8721 | 0.8704 | 0.8699 |
| auc_2h | 0.8775 | 0.8811 | 0.8787 | 0.8794 |
| rmse_hours | 10.71 | 10.89 | 10.76 | 10.98 |
| mae_hours | 4.16 | 4.32 | 4.23 | 4.38 |
| n_alleles | 73 | 73 | 73 | 73 |

Reproduced. Reference numbers land just below my three repeats — consistent with a torch
version difference (2.14.1 here vs the >=2.6.0 pin), not a logic difference.

**Methodological finding that governs every comparison below.** All three runs above used
`--seed 42` and the *same* split, yet differ. `train_baseline.py` calls `np.random.seed(seed)`
but **never calls `torch.manual_seed`**, so the seed fixes the data split only — weight
initialisation and batch shuffling are unseeded. Observed run-to-run spread at fixed seed and
fixed split:

- `global_pcc_score`: 0.7190 – 0.7242, spread **0.005**
- `mean_allele_pcc`: 0.5259 – 0.5323, spread **0.006**
- `rmse_hours`: 10.76 – 10.98, spread **0.22**

**Any claimed improvement smaller than ~0.005 global PCC is indistinguishable from this
noise floor.** My harness seeds torch explicitly so that seed-to-seed spread reflects a
controlled resample rather than uncontrolled nondeterminism.

### All three splits, seed 42

| split | train/val/test | mean_allele_pcc | global_pcc | auc_1h | rmse_h | n_alleles |
|---|---|---|---|---|---|---|
| random | 20278 / 2254 / 5634 | 0.5300 | 0.7242 | 0.8721 | 10.89 | 73 |
| cluster (peptide-grouped) | 19984 / 2221 / 5961 | 0.5069 | 0.6842 | 0.8507 | 10.31 | 72 |
| allele (leave-allele-out) | 19100 / 2123 / 6943 | 0.3614 | 0.4653 | 0.7512 | 8.07 | 15 |

Difficulty ordering is as expected and large: **random 0.724 > cluster 0.684 >> allele 0.465**
global PCC. Leave-allele-out (15 held-out alleles) is far harder and is the split that
actually tests the premise — a *per-allele* HLA embedding can only help there if it
generalises to alleles never seen in training. On random and cluster splits every test allele
appears in training, so any per-allele vector can act as a memorised lookup key.

That observation dictates an extra control the brief did not ask for: an **allele one-hot**
HLA encoding. If Boltz-2 matches one-hot on the random split, then whatever it contributes is
allele *identity*, not structural information — and only the leave-allele-out split can tell
those apart.
