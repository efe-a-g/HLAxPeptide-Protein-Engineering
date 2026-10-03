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

---

## Entry 6 — DECISIVE: the HLA-side gain does not survive proper training

This is the result the whole sprint turns on. Phase 1 found that replacing the BLOSUM
pseudosequence with Boltz-2 pocket embeddings gained +0.032 mean per-allele PCC on the random
split (t = +4.1), and ESM2-HLA +0.018 (t = +2.9). Both were measured at the published
25-epoch protocol, which Entry 5 showed to be **undertrained**. Re-running the same arms at
100 epochs:

### Random split, mean per-allele PCC (5 seeds, paired per seed)

| HLA representation | 25 epochs | Δ vs BLOSUM | 100 epochs | Δ vs BLOSUM |
|---|---|---|---|---|
| BLOSUM pseudosequence | 0.5362 | — | **0.6293** | — |
| Boltz-2 pockets (B+F) | 0.5683 | **+0.0321** (t=+4.05) * | 0.6141 | **−0.0152** (t=−2.13) |
| ESM-2 HLA (mean-pooled) | 0.5546 | **+0.0184** (t=+2.87) * | 0.6086 | **−0.0208** (t=−7.33) * |
| allele one-hot (control) | 0.2678 | −0.2684 (t=−34.1) * | 0.2764 | −0.3530 (t=−33.8) * |

**The sign flips.** On global PCC the reversal is unambiguous for both: Boltz −0.0104
(t = −10.9) and ESM2-HLA −0.0097 (t = −4.7), both significant losses at 100 epochs after
being significant gains at 25.

### All three splits at 100 epochs, Δ mean per-allele PCC vs BLOSUM-HLA

| split | Boltz-HLA | ESM2-HLA |
|---|---|---|
| random | −0.0152 (t=−2.13, n.s.) | **−0.0208 (t=−7.33) \*** |
| cluster | −0.0014 (t=−0.15, n.s.) | +0.0043 (t=+0.44, n.s.) |
| allele | **−0.0548 (t=−2.79) \*** | — |

Nowhere does a learned HLA representation beat the pseudosequence once the baseline is
trained to convergence. It is **neutral at best** (peptide-grouped split, Δ ≈ 0) and
**significantly worse** on the random and leave-allele-out splits.

### Interpretation

The phase-1 gain was **a convergence artifact, not information**. A 768-dim z-scored dense
block is an easier optimisation target than a 680-dim sparse BLOSUM block, so it reaches a
good solution in fewer epochs. Given 4x the epochs the BLOSUM baseline overtakes it and keeps
going. This is precisely the failure mode that a fixed, short training budget manufactures —
and it would have been reported as a positive result by any study that fixed epochs at the
published value and did not check.

**I would have published the wrong answer had I stopped after phase 1.** The single most
valuable thing in this log is the 100-epoch control, which cost 45 runs and 25 minutes.

### What stands, and what does not

Still standing from phase 1:

- **The one-hot control.** Allele identity alone is catastrophic (0.276 vs 0.629 at 100
  epochs). The pseudosequence's value is the *similarity structure* it provides across
  alleles, not identity. Boltz-2 simply does not improve on that structure.
- **Global PCC is inflated.** An HLA-only arm that never sees the peptide reaches 0.543
  global PCC. Mean per-allele PCC remains the honest metric.
- **ESM-2 peptide features lose decisively** (−0.19 at 100 epochs), and the gap widens with
  training rather than closing.

Withdrawn:

- "Boltz-2 pocket embeddings provide a better similarity space than the BLOSUM
  pseudosequence." **False at convergence.** True only against an undertrained baseline.

### Answer to the organisers' question, as it currently stands

*Are open protein foundation models useful for predicting peptide-HLA class I stability?*
**On this dataset, with frozen features and a small supervised head: no.** The HLA side gains
nothing once the baseline is trained properly; the peptide side is actively harmed. The
single largest improvement found anywhere in this sprint came from training the *existing*
baseline four times longer (+0.093 mean per-allele PCC) — three times what any foundation-model
feature offered, and free.

---

## Entry 7 — Phase 3 complete: the full ablation (330 runs, 0 failures)

All deltas are **mean per-allele PCC vs the BLOSUM baseline at the same epoch budget**,
paired per seed, 5 seeds. `*` = |t| > 2.5. Columns: random / peptide-grouped / leave-allele-out.

### Peptide encoder (25 epochs)

| peptide features | dims | random | cluster | allele |
|---|---|---|---|---|
| ESM-2 150M per-residue | 5760 | −0.1062 * | −0.1859 * | −0.0032 |
| **ESM-2 150M PCA-180** | 180 | **−0.0265** | −0.0936 * | **+0.0065** |
| ESM-2 150M mean-pooled | 640 | −0.1192 * | −0.2474 * | −0.0649 * |
| ESM-2 35M per-residue | 4320 | −0.0894 * | −0.1648 * | −0.0053 |

Three things fall out:

1. **Dimensionality was a large part of the penalty, not the representation.** Compressing
   ESM-2 to BLOSUM width recovers +0.080 (random) and leaves the gap statistically
   indistinguishable from the baseline there and on leave-allele-out. It remains clearly
   worse on the peptide-grouped split — the split that specifically tests unseen peptides.
2. **Mean pooling is worse than per-residue, everywhere.** For 9-mers this is expected and
   worth stating: peptide-MHC binding is driven by anchor residues at P2 and the C-terminus,
   and averaging over 9 positions destroys exactly that. Anyone mean-pooling a PLM over a
   short peptide is discarding the signal.
3. **The bigger model is not better.** ESM-2 35M beats ESM-2 150M on every split
   (−0.089 vs −0.106 random). No scaling benefit at this task size.

### HLA encoder — replace the pseudosequence

| | random | cluster | allele |
|---|---|---|---|
| Boltz-2 B+F, **25 ep** | **+0.0321** * | **+0.0338** * | −0.0147 |
| Boltz-2 B+F, **100 ep** | −0.0152 | −0.0014 | **−0.0548** * |
| ESM2-HLA, **25 ep** | **+0.0184** * | **+0.0342** * | −0.0558 * |
| ESM2-HLA, **100 ep** | −0.0208 * | +0.0043 | −0.1216 * |
| allele one-hot, 25 ep | −0.2684 * | −0.2568 * | −0.1202 * |

### HLA encoder — augment (keep pseudosequence, add learned block)

| | random | cluster | allele |
|---|---|---|---|
| BLOSUM + Boltz-2, **25 ep** | **+0.0313** * | **+0.0371** * | −0.0089 |
| BLOSUM + Boltz-2, **100 ep** | −0.0112 | +0.0003 | −0.0554 * |
| BLOSUM + ESM2-HLA, **25 ep** | +0.0138 | +0.0171 | −0.0561 * |
| BLOSUM + ESM2-HLA, **100 ep** | −0.0247 * | −0.0012 | −0.0978 * |

**Augmentation behaves exactly like replacement**: a clear gain at 25 epochs, gone at 100.
This matters because augmentation is the configuration anyone would actually ship — "keep what
works, add the embedding" — and it is subject to the identical artifact.

### And the intervention that actually works

| | random | cluster | allele |
|---|---|---|---|
| **unchanged baseline, 100 ep vs 25 ep** | **+0.0931** * | **+0.0780** * | **+0.0661** * |

## Final answer to the organisers' question

> *Are open protein foundation models useful for predicting peptide-HLA class I stability?*

**On this dataset, with frozen embeddings and a small supervised head: no.**

- **HLA side.** Boltz-2 pocket embeddings and ESM-2 HLA embeddings appear to help (+0.03) at
  the published 25-epoch protocol, whether replacing or augmenting the pseudosequence. The
  effect **reverses once the baseline is trained to convergence** and is negative or neutral
  everywhere at 100 epochs. It was optimisation speed, not information.
- **Peptide side.** ESM-2 never beats BLOSUM50. At matched dimensionality it draws level on
  two splits and loses on the one that tests unseen peptides. Mean pooling makes it worse; a
  bigger ESM-2 makes it worse.
- **The honest headline is not about foundation models at all.** The largest, cheapest and
  most reliable improvement found anywhere in 330 runs was training the existing baseline four
  times longer: **+0.093 / +0.078 / +0.066** mean per-allele PCC across the three splits, all
  significant — roughly three times what the best foundation-model feature ever delivered, and
  it costs nothing but epochs.

### Limits of this conclusion — what would change it

1. **Frozen features only.** No fine-tuning was tested; that is where PLMs usually earn their
   keep, and it needs a GPU this machine does not have (Intel Arc iGPU, no CUDA).
2. **Boltz-2 embeddings are per-allele** — 75 vectors for 28,166 rows. They cannot represent
   peptide-specific structure by construction. A **per-complex** embedding (one structure per
   peptide-HLA pair) is a genuinely different and untested experiment, and is the single most
   promising follow-up.
3. **Leave-allele-out is underpowered.** Baseline seed SD there is 0.081, 61x the random
   split; 15 held-out alleles per seed means which alleles are drawn matters more than the
   model. Proper leave-one-allele-out CV over all 75 alleles is needed to settle the
   unseen-allele case, and nothing here should be read as settling it.
4. **One architecture.** A 60-unit MLP; results may differ for a model with capacity to
   exploit 5,760-dim inputs — though the 100-epoch and PCA probes both argue the input
   representation, not capacity, is the binding constraint.

---

## Entry 8 — Deliverables, and how to regenerate everything

### The report

`outputs/report/index.html` — open it directly in a browser, no server needed. Self-contained:
all 330 runs are embedded as JSON, charts are hand-built SVG drawn in vanilla JS, and both
NGL (`ngl.js`) and the six PDB coordinate files (`structures/`) are bundled alongside.

**Why bundled rather than CDN.** The brief suggested loading NGL and structures from a CDN.
NGL itself loads fine that way, but its `rcsb://` datasource **failed to fetch coordinates
from a `file://` origin** — verified by a standalone smoke test, which returned
`error loading file: 'network error'` and would have left every 3D panel blank for anyone
opening the page. Bundling makes the report work offline and removes the failure mode. The
RCSB usage policy is linked in the page and recorded in `.licenses/pdb_database_LICENSE.txt`.

Contents: verdict tiles showing the 25-epoch vs 100-epoch sign flip; the noise-floor table;
per-split ablation bars with a metric selector (mean per-allele PCC default, global PCC, AUC,
RMSE) and a training-protocol selector (published / 100 epochs / PCA-180); paired-delta charts
with the noise floor drawn as a shaded band; per-allele dot plots; six interactive 3D
complexes; and table views of everything for accessibility.

Palette validated for colour-vision deficiency in both light and dark mode with the dataviz
validator (3 slots, all-pairs, worst CVD delta-E 9.2 light / 9.4 dark against a >= 8 target).
Rendered and visually checked by headless-Chrome screenshot rather than assumed — which is how
the blank-3D-panel bug was caught.

### Verification performed, not assumed

| check | result |
|---|---|
| embedding parquets load, row order matches CSV | yes, after binary re-extraction |
| Boltz embeddings are per-allele | 75 distinct rows / 28,166 |
| cluster split peptide leakage | 0 shared, all 5 seeds |
| allele split allele leakage | 0 shared, all 5 seeds |
| baseline reproduces committed metrics | yes, within init noise |
| grid failures | 0 of 330 |
| report JSON parses, charts/tables/viewers render | yes, by screenshot |
| 3D coordinates load | yes, from bundled local files |

### Regenerate

```
bash scripts/extract_embeddings.sh      # binary-safe; verifies PAR1 head and tail
uv run python python/verify_data.py     # dataset + embedding facts
uv run python python/verify_splits.py   # leakage audit
uv run python python/embed_esm.py       # ESM-2 cache (minutes, CPU)
uv run python python/run_grid.py        # phase 1: 135 runs, ~26 min
uv run python python/run_phase2.py --probes noise capacity
uv run python python/run_phase3.py      # 150 runs, ~70 min
uv run python python/noise_floor.py
uv run python python/aggregate.py       # console tables + CSVs
uv run python python/build_report.py    # outputs/report/index.html
```

### Known gaps, stated rather than hidden

- The 256-unit-head probe (`h256`) was stopped after 7 runs and only its baseline completed,
  so it is excluded from the report. The hypothesis it tested was answered by the 100-epoch
  probe; it is not a missing result so much as an abandoned duplicate.
- Leave-allele-out remains underpowered (seed SD 0.081, 61x the random split). Proper
  leave-one-allele-out CV over all 75 alleles is the correct experiment and was not run.
- No fine-tuning of any foundation model was attempted — no CUDA on this machine.

---

## Entry 9 — Report switched to Spearman, and made a single shareable file

### Spearman is now the headline metric

The report defaults to **mean per-allele SCC**; PCC, global SCC, global PCC, AUC and RMSE
remain selectable. Rationale, which is worth stating because it is not merely taste:

- **Per-allele rather than global** — an HLA-only arm that never sees the peptide reaches
  ~0.54 global correlation. Global is dominated by between-allele differences in mean
  stability, so it rewards modelling the allele marginal rather than the interaction.
- **Spearman rather than Pearson** — 20.2% of the measurements sit at t-half exactly 0 and the
  rest is heavily right-skewed (median 1.1 h, max 256.7 h). Pearson therefore depends on which
  scale you compute it on: **0.540 on the transformed score vs 0.333 on raw hours**, same runs,
  a 0.22 gap. Spearman has no such ambiguity, and I verified it is invariant to the target
  transform here: `global_scc_score` and `global_scc_thalf` agree to **max |difference|
  0.000012** across 182 runs. So the reported SCC *is* the ordering of half-lives in hours.

Every conclusion is unchanged under SCC, which is itself worth recording — the findings are
not an artifact of the correlation measure:

| | PCC | SCC |
|---|---|---|
| baseline, random, 25 ep | 0.5362 | 0.5333 |
| baseline, random, 100 ep | 0.6293 | 0.6302 |
| Boltz-HLA delta, 25 ep | +0.0321 * | +0.0372 * |
| Boltz-HLA delta, 100 ep | −0.0152 | −0.0142 |
| training longer, random | +0.0931 * | +0.0969 * |

### Two viewer bugs found by actually opening the page

1. **The 3D panels were blank on a normal double-click.** Chrome refuses a `file://` page
   permission to read sibling `file://` resources, so `loadFile("structures/<id>.pdb")` failed
   silently. My first check missed it because the headless run passed
   `--allow-file-access-from-files`, which grants exactly the permission a real double-click
   lacks — **I had verified under conditions friendlier than the user's.** Fixed by embedding
   the coordinates and loading them from an in-memory Blob.
2. **Sharing the HTML alone would have broken it**, since `ngl.js` was a sibling `<script src>`.
   Now inlined.

To keep the page affordable with coordinates embedded, each PDB is reduced to what is actually
drawn: the three rendered chains only, first model, primary altloc, no ANISOU records, no
waters, no header metadata. 4.4 MB of raw PDB becomes a 2.9 MB single file carrying all six
structures *and* the viewer library.

`outputs/report/index.html` is now **one self-contained file** — no sibling files, no network,
no server. `--no-inline-ngl` restores the split layout if ever wanted.

---

## Entry 10 — CORRECTION: my splits were not congruent with the repo convention

`origin/baseline` advanced to **ed9d863 "update baseline code"** after this worktree was cut.
It deletes `train_baseline.py` and replaces it with `baseline.py`, which states a binding
project-wide convention:

> *"this applies to every model in this repo, not just the baseline. Later models
> (foundation-model embeddings, fine-tuned heads, etc.) should import `split_by_supertype`
> and `evaluate` from here and use them unchanged, so results are comparable across
> approaches."*

**Everything in Entries 1-9 predates that commit and does not comply.** The arithmetic is
sound and the internal comparisons are valid, but the numbers are not comparable with anything
else in the repo. Divergences:

| | `baseline.py` @ ed9d863 | Entries 1-9 |
|---|---|---|
| split | supertype-stratified leave-allele-out, one canonical split | random / peptide-grouped / naive allele holdout |
| split unit | **unique pseudosequence** | allele name |
| test_size | 0.25 | 0.20 |
| test eligibility | **>= 100 measurements**; Unclassified and singleton supertypes forced to train | none |
| epochs | **90**, no validation split, no early stopping | 25 (then 100 in my probes) |
| metric | mean per-allele Spearman | same (reached independently) |
| null model | `peptide_only_null` | equivalent arm present |

Two points of accidental agreement are worth recording: the team standardised on **mean
per-allele Spearman**, which I had switched to independently, and on **90 epochs**, which
independently corroborates Entry 5's finding that 25 was undertrained.

Their split is **better than mine** in two specific ways I should have anticipated:

1. Splitting on pseudosequence, not allele name. I noted in Entry 0 that C67S constructs are
   invisible to the pseudosequence encoding, then failed to act on it — my allele split could
   place a model-identical allele on both sides.
2. The >= 100-measurement floor. This is almost certainly the main cause of the pathological
   variance in Entry 4: my split could hold out an allele with 16 rows, whose noise-level
   Spearman then carried **equal weight** in the mean. Under the canonical split the smallest
   held-out allele has 350 measurements, and the baseline seed SD falls from **0.081 to 0.038**.

### Re-run under the canonical protocol (`python/run_supertype.py`, 40 runs)

Imports `split_by_supertype`, `evaluate`, `transform_target`, `encode_sequences`,
`train_model` and `peptide_only_null` from `baseline.py` unchanged; only input features vary.
Split: 21,943 train / 6,223 test, 16 held-out alleles across 10 supertypes, smallest 350 rows.

| arm | dims | mean per-allele SCC | paired Δ vs baseline |
|---|---|---|---|
| **BLOSUM-pep \| BLOSUM-HLA (baseline)** | 860 | **0.4626 ± 0.0379** | — |
| ESM2-pep \| Boltz-HLA | 6528 | 0.4206 ± 0.0424 | −0.0420 (t=−3.17) * |
| BLOSUM-pep \| BLOSUM+Boltz (augment) | 1628 | 0.4183 ± 0.0271 | −0.0442 (t=−4.99) * |
| BLOSUM-pep \| Boltz-HLA (replace) | 948 | 0.3974 ± 0.0282 | −0.0651 (t=−14.75) * |
| ESM2-pep \| BLOSUM-HLA | 6440 | 0.3914 ± 0.0356 | −0.0712 (t=−6.32) * |
| BLOSUM-pep \| ESM2-HLA | 820 | 0.3151 ± 0.0436 | −0.1474 (t=−12.13) * |
| BLOSUM-pep \| onehot-HLA (control) | 255 | 0.3148 ± 0.0236 | −0.1477 (t=−8.23) * |
| *peptide-only null (no allele info)* | 0 | *0.3045 ± 0.0163* | −0.1581 (t=−9.99) * |

**Every foundation-model arm loses to the BLOSUM baseline, every one outside the noise floor.**
No ambiguity, no sign flip, nothing inside noise — a cleaner negative than Entries 6-7 produced.

Two further readings:

- **The margin allele modelling buys is +0.158** (baseline 0.4626 vs peptide-only null 0.3045,
  t=+9.99). That is the budget any HLA representation is competing for.
- **ESM-2 on the HLA side (0.3151) is statistically indistinguishable from an allele one-hot
  (0.3148) and barely above the peptide-only null (0.3045).** A mean-pooled PLM embedding of
  the alpha1/alpha2 domain carries essentially no usable allele information beyond identity.

### Status of the earlier entries

Retained, not deleted. They are a valid internal ablation and contain the methodological
findings (noise floor, global-PCC inflation, the convergence artifact, the one-hot control,
dimension-matching). But **the canonical-split table above is the result that should be
quoted**, and the report now leads with it; the earlier splits are kept as supporting context
behind the protocol selector.
