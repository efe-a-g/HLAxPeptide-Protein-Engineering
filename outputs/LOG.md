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

---

## Entry 2 — Ablation design, and the two traps it has to avoid

Everything from here uses `python/run_ablation.py`, which **imports** the model,
loss, target transform, metrics and split logic from `train_baseline.py` rather than
reimplementing them. Only the input features differ between arms: identical 60-unit
sigmoid MLP, MSE on `s = 2^(-t0/thalf)`, Adam lr 1e-3, wd 1e-5, batch 128, 25 epochs,
best-validation checkpoint. It also calls `torch.manual_seed`, which the baseline does not.

### Feature blocks

| block | dims | what it is |
|---|---|---|
| `BLOSUM-pep` | 9x20 = 180 | BLOSUM50/5, per position |
| `BLOSUM-HLA` | 34x20 = 680 | NetMHCpan pseudosequence (= baseline HLA side) |
| `ESM2-pep` | 9x640 = 5760 | ESM-2 t30_150M per-residue — positional analogue of BLOSUM |
| `ESM2-HLA` | 640 | ESM-2 t30_150M mean-pooled over the 182-aa alpha1/alpha2 domain |
| `Boltz-HLA(B+F)` | 2x384 = 768 | Boltz-2 pocket embeddings, B pocket + F pocket |
| `onehot-HLA` | 75 | allele indicator — **the control that makes the result interpretable** |

ESM-2 was run on CPU over the 5,633 unique peptides and 75 unique HLA domains, cached
to `cache/` keyed by sequence. ~55 s for the peptides at t30_150M; nothing is recomputed.

### Trap 1 — standardization is not cosmetic

Boltz features reach magnitude ~1000 (mean L2 norm 1656) against BLOSUM's ~[-1, 3].
Raw, they saturate the sigmoid first layer and nothing trains. All learned blocks are
z-scored on **train-only** statistics. BLOSUM is left raw so the baseline arm continues
to match `train_baseline.py` exactly. Skipping this step alone would have produced a
confident, wrong negative result.

### Trap 2 — a per-allele vector is a lookup key, not knowledge

The Boltz embeddings have 75 distinct values. On the random and peptide-grouped splits
*every test allele also appears in training*, so a per-allele vector can be memorised as
an index. That is why `onehot-HLA` is in the grid: it is the ceiling for "allele identity
and nothing else". Only leave-allele-out can separate structural information from identity.

### Split audit (`python/verify_splits.py`)

pandas 3.0 returns an `ArrowStringArray` from `.unique()`, and NumPy warns that shuffling
such an object "may contain duplicates after shuffling". `train_baseline.py`'s cluster
split depends on exactly that shuffle, so a silent failure there would have invalidated the
peptide-grouped split. Audited all 3 strategies x 5 seeds:

- `cluster`: **0 peptides shared** between train and test, every seed. Clean.
- `allele`: **0 alleles shared**, 15 held-out alleles per seed. Clean.
- `random`: ~2,770 peptides shared — i.e. **~49% of test peptides are also in training**
  (paired with different alleles). This is why random scores above cluster, and it is a
  property of the split, not of any model.
- Row coverage exact, no train/test row overlap anywhere.

Verdict: the warning is benign in this version. Documented rather than assumed.

---

## Entry 3 — Phase 1 results: 9 configurations x 3 splits x 5 seeds (135 runs, 0 failures, 26 min)

Full tables in `outputs/ablation/summary_mean_std.csv` and `summary_deltas.csv`.
Deltas are **paired per seed** against the baseline, then averaged, so both arms see the
same split; `t` is mean delta / standard error over 5 seeds. I treat |t| > 2.5 as outside
the noise floor.

### Random split — mean per-allele PCC (the honest metric, see below)

| configuration | dims | mean allele PCC | global PCC | Δ global PCC vs baseline |
|---|---|---|---|---|
| BLOSUM-pep + **Boltz-HLA(B+F)** | 948 | **0.5683** ± 0.0221 | 0.7375 | **+0.0160** (t=+4.0) |
| BLOSUM-pep + **ESM2-HLA** | 820 | 0.5546 ± 0.0174 | 0.7331 | **+0.0115** (t=+6.6) |
| BLOSUM-pep + BLOSUM-HLA *(baseline)* | 860 | 0.5362 ± 0.0118 | 0.7216 | — |
| ESM2-pep + Boltz-HLA | 6528 | 0.5097 | 0.7133 | −0.0083 (t=−4.7) |
| ESM2-pep + ESM2-HLA | 6400 | 0.4869 | 0.6962 | −0.0254 (t=−10.4) |
| ESM2-pep + BLOSUM-HLA | 6440 | 0.4300 | 0.6411 | −0.0804 (t=−16.9) |
| BLOSUM-pep + **onehot-HLA** | 255 | **0.2678** | 0.5969 | −0.1247 (t=−56.7) |
| **none** + BLOSUM-HLA | 680 | *n/a* | **0.5433** | −0.1782 (t=−59.2) |
| BLOSUM-pep + none | 180 | 0.2092 | 0.3615 | −0.3600 (t=−77.7) |

### Finding 1 — the one-hot control refutes the "memorised lookup" explanation

My prior was that Boltz-2's gain on the random split would just be allele identity, since
the embeddings are 75 distinct vectors and every test allele is in training. **That is wrong.**
An allele one-hot — the purest possible identity encoding — scores mean-allele-PCC **0.268**
against BLOSUM's 0.536 and Boltz's 0.568. A lookup key is far *worse* than the pseudosequence.

So the pseudosequence's value is not identity but *similarity*: it lets the model share
statistical strength across related alleles. Boltz-2 pocket embeddings provide a **better
similarity space than BLOSUM pseudosequence** — a real, if small, positive result.
Without the one-hot control I would have reported the opposite conclusion with confidence.

### Finding 2 — global PCC is inflated and should not be the headline

`none + BLOSUM-HLA` sees **no peptide information whatsoever** and still reaches global PCC
**0.5433** — 75% of the baseline's 0.7216. It can only predict one number per allele, so all
of that correlation is between-allele variation in mean stability. Its mean per-allele PCC is
undefined (constant prediction within an allele → zero variance → excluded by the metric),
which is exactly right and exactly the point.

Meanwhile peptide-only reaches 0.3615. **The HLA side alone out-scores the peptide side alone
on global PCC**, which inverts the biological importance of the two. Any leaderboard built on
global PCC rewards modelling the allele marginal, not peptide-HLA interaction.
Mean per-allele PCC is reported as primary from here on.

### Finding 3 — the gain is real but confined to seen alleles

| configuration | random | cluster | **allele (unseen)** |
|---|---|---|---|
| Boltz-HLA, Δ global PCC | +0.0160 (t=+4.0) * | +0.0126 (t=+6.3) * | **−0.1771 (t=−2.4, n.s.)** |
| ESM2-HLA, Δ global PCC | +0.0115 (t=+6.6) * | +0.0148 (t=+5.5) * | **−0.1442 (t=−2.5)** |
| Boltz-HLA, Δ mean allele PCC | +0.0321 | +0.0338 | **−0.0147 (n.s.)** |

Both learned HLA representations help consistently when test alleles are seen in training,
and both degrade on unseen alleles. The leave-allele-out deficits are *large* but sit at
|t| ≈ 2.4, i.e. **not formally significant**, because that split's seed variance is enormous
(global PCC SD ±0.08 to ±0.17 across seeds, against ±0.001 to ±0.013 on random). With 15
held-out alleles per seed, which alleles get held out matters more than the model does.
Honest statement: on unseen alleles I cannot demonstrate that the learned HLA features help,
and the point estimates say they hurt.

A sharper version of the same contrast: on leave-allele-out, Boltz's **global** PCC collapses
(0.285 vs 0.463) while its **per-allele** PCC barely moves (0.297 vs 0.312, Δ −0.015 n.s.).
It loses the absolute stability *scale* for an unseen allele while still ranking peptides
within it. That is the specific failure mode, and global PCC alone would have hidden it.

### Finding 4 — ESM-2 peptide features lose badly, cause not yet established

Replacing BLOSUM-pep with per-residue ESM-2 t30_150M costs −0.080 global PCC on random and
−0.119 on cluster, both far outside noise. But the ESM arm feeds 5,760 dims into the same
60-unit head on the same 25-epoch budget as a 180-dim block, so this could be underfitting
rather than a worse representation. **Not reporting a verdict on ESM-2 peptide features until
the capacity probe lands** (phase 2: 100 epochs, 256-unit head, and PCA to 180 dims).

### Carried forward / discarded

- Discarded the "Boltz = allele lookup" hypothesis — refuted by the one-hot control.
- `mean_allele_pcc` is NaN for HLA-only by construction, not a bug. Reported as n/a.
- Phase 2 launched: noise floor (split fixed, torch seed varied), capacity probes,
  pooling (mean vs per-residue, 35M vs 150M), and **augmentation** (pseudosequence *plus*
  learned block, rather than replacement) — the configuration that matters in practice.

---

## Entry 4 — The noise floor, measured three ways (`python/noise_floor.py`)

| source of variation | runs | mean global PCC | SD | range |
|---|---|---|---|---|
| Unseeded published script (seed 42, split fixed) | 3 | 0.7209 | 0.0029 | 0.0052 |
| Initialisation only (split fixed, `--torch_seed` varied) | 8 | 0.7208 | **0.0028** | 0.0072 |
| Seed varied (split + init), random split | 5 | 0.7216 | 0.0013 | 0.0029 |

**Floor adopted: SD 0.0029 global PCC.** The controlled initialisation-only measurement
(0.0028) agrees closely with the uncontrolled repeats of the published script (0.0029),
which is reassuring — the unseeded nondeterminism in `train_baseline.py` is init noise and
nothing stranger.

### Correction to an earlier assumption

I had expected split variance to dominate initialisation variance. On the random split it is
the **opposite**: seed-to-seed SD is 0.0013 versus 0.0028 for init alone. A random 80/20 split
of 28,166 rows is extremely stable, so re-drawing it changes almost nothing and the residual
spread is weight init. Worth stating plainly because it inverts the usual intuition.

### But the splits are not comparable in this respect at all

Baseline arm, seed spread of global PCC per split:

| split | mean | SD | range over 5 seeds |
|---|---|---|---|
| random | 0.7216 | 0.0013 | 0.0029 |
| cluster | 0.7028 | 0.0133 | 0.0313 |
| **allele** | **0.4625** | **0.0812** | **0.2165** |

**Leave-allele-out seed SD is 61x the random-split seed SD.** The baseline's global PCC on
that split ranges from roughly 0.37 to 0.58 depending only on which 15 alleles were held out.

Consequences, which govern how every leave-allele-out claim on this project is worded:

1. A **single-seed** leave-allele-out number is nearly meaningless. Any paper or leaderboard
   reporting one should be treated with suspicion.
2. Arms must be compared **paired within a seed** (same held-out alleles) and the differences
   averaged. An unpaired comparison of means across seeds would be swamped.
3. Even paired, 5 seeds on this split gives |t| ~ 2.4 for a point-estimate difference as large
   as -0.18 global PCC. The honest conclusion for unseen alleles is therefore
   **"not demonstrated either way"**, not "foundation models fail" — the experiment as designed
   lacks the power to resolve it. More seeds, or proper leave-one-allele-out cross-validation
   over all 75 alleles, would be the fix. Flagged as the main power limitation of this sprint.

---

## Entry 5 — Capacity probe: the ESM-2 peptide loss is real, and the published protocol is undertrained

Phase 2 asked whether ESM-2's peptide-side deficit was a worse representation or merely an
underfitted 5,760-dim input in a 60-unit head on a 25-epoch budget. Answer: **a worse
representation**, and the 25-epoch budget was *flattering* to it.

Mean per-allele PCC, BLOSUM-HLA on the HLA side throughout, 5 seeds:

| peptide features | 25 epochs | 100 epochs | change |
|---|---|---|---|
| BLOSUM-pep (baseline) | 0.5362 | **0.6293** | **+0.093** |
| ESM2-pep (t30_150M, per-residue) | 0.4300 | 0.4390 | +0.009 |
| **paired Δ (ESM − BLOSUM)** | **−0.1062** (t=−12.4) | **−0.1904** (t=−21.3) | gap widens |

Same direction on every split (cluster Δ −0.263, allele Δ −0.064, all |t| > 5). Four times the
training budget buys BLOSUM **ten times** what it buys ESM-2. Hypothesis "ESM-2 peptide features
were just undertrained" is **discarded**.

### The more consequential finding, which is not about foundation models at all

**The published 25-epoch protocol is undertrained.** Training the *unchanged* baseline for
100 instead of 25 epochs:

| split | mean allele PCC 25ep → 100ep | global PCC 25ep → 100ep |
|---|---|---|
| random | 0.5362 → **0.6293** (+0.093) | 0.7216 → 0.7868 (+0.065) |
| cluster | 0.4943 → **0.5723** (+0.078) | 0.7028 → 0.7565 (+0.054) |
| allele | 0.3115 → **0.3777** (+0.066) | 0.4625 → 0.4899 (+0.027) |

**Training longer improves the baseline roughly 3x more than the best foundation-model feature
does** (+0.093 vs +0.032 mean allele PCC on random). The cheapest available win on this problem
is epochs, not embeddings. Any comparison run only at 25 epochs — including all of my phase 1 —
is a comparison against a handicapped baseline.

### Consequence: phase 2 was stopped and re-prioritised

This invalidates the *framing* of phase 1, though not its arithmetic: the HLA-side gains
(+0.032 Boltz, +0.018 ESM2-HLA) were measured against an undertrained baseline, and a feature
that merely helps a model converge faster will show exactly that signature. The urgent question
became **does the HLA-side gain survive at 100 epochs?**

Actions taken:

- **Killed the 256-unit-head probe** (`h256`) after 2 of 30 runs. It tested the same
  underfitting hypothesis the 100-epoch probe had already answered, and it was the slowest
  stage in the queue. Its 2 completed runs are retained in `results.jsonl` but excluded from
  the report, which now drops any variant with fewer than 3 seeds rather than show a 2-seed
  arm beside 5-seed arms.
- **Launched phase 3** (`python/run_phase3.py`) ordered by value, so the most important result
  lands first: (1) HLA-side arms at 100 epochs on all splits, (2) PCA-180 dimension-matched
  ESM-2 peptide, (3) augmentation (pseudosequence *plus* learned block) at 25 and 100 epochs,
  (4) pooling and model-size comparison.

Timing note: phase 2's ETA estimates were badly wrong (predicted ~38 min, ran ~80 min) because
the 100-epoch and wide-head runs are far slower than the 25-epoch runs the average was seeded
from. Recorded so later estimates are not trusted blindly.
