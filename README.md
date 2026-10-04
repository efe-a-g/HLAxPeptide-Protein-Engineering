# Peptide–HLA stability with protein foundation models

*Serova Protein Engineering Track — London AI × Science Hackathon, 3–4 October 2026*

**The question the organisers asked:** are open protein foundation models useful for
predicting the stability (half-life) of peptide–HLA class I complexes?

**The short answer this repo arrived at:** mostly no, with one exception.

- Frozen **Boltz-2 embeddings of the HLA groove** do not beat the BLOSUM pseudosequence. ✗
- **Boltz-2's own confidence** in a co-folded complex carries real but far too little signal. ✗
- **Inverse-folding likelihoods** (LigandMPNN) *do* help, as extra features on top of the
  sequence baseline. ✓
- The largest single improvement found anywhere was **training the existing baseline longer** —
  no foundation model involved.

A well-supported negative result is a valid answer to "are they useful", and three of the four
workstreams report one. The numbers below are stated with the protocol they were measured
under, because they are **not** interchangeable across workstreams.

---

## The data

`rasmussen_et_al_dataset.csv` — Rasmussen et al. (2016), *J Immunol* 197(4):1517–1524.

| property | value |
|---|---|
| measurements | 28,166 |
| alleles | 75 (74 unique pseudosequences) |
| unique peptides | 5,633 |
| peptide length | 9 only |
| columns | `allele`, `peptide`, `thalf_hours`, `hla_seq` (182 aa, α1/α2), `hla_pseudoseq` (34 aa) |
| target transform | `s = 2^(-t0 / t_half)`, `t0` = 1 h |

Two properties that shaped every experiment:

1. **20.2% of rows have `thalf_hours` exactly 0.** A fifth of the target is a point mass at the
   boundary. This caps achievable correlation and makes RMSE-in-hours a poor headline metric.
2. **Three engineered C67S constructs are invisible to the pseudosequence encoding.** Position 67
   is not among the 34 NetMHCpan contact residues, so a C67S mutant cannot be distinguished from
   its wild type by the baseline — and splitting on *allele name* rather than *pseudosequence*
   leaks between them.

## The shared protocol

`python/baseline.py` (on the `baseline` branch) states a project-wide convention that later
models are expected to import rather than reimplement:

- **Split** — hold out whole alleles, stratified by HLA supertype (Sidney et al. 2008) so every
  supertype appears in both train and test. This measures generalisation to a *new allele within
  a known motif family*, which is the personalised-immunotherapy case.
- **Metric** — **mean per-allele Spearman** over held-out alleles. We care about ranking
  candidate peptides for a fixed allele, not absolute hours.

Two reference points recur throughout:

- the **BLOSUM baseline** — 9-mer + 34-aa pseudosequence, BLOSUM50/5, 43 × 20 = 860 dims → MLP;
- the **peptide-only null** — the same model with the HLA thrown away entirely.

The margin between them (~0.15 Spearman) is the entire budget any HLA representation competes for.

> ⚠️ **The baseline's own score moved during the sprint** (≈0.40 → 0.47) as epochs, validation and
> architecture were revised, and two workstreams predate the canonical split. Always read a delta
> against the baseline *in the same table*, never across tables.

---

## Repository map

Work was split across branches rather than merged, so each branch is a self-contained workstream.

| branch | workstream | what's on it |
|---|---|---|
| `main` | **presentation** | the slide decks and this README |
| `baseline` | **sequence baseline + inverse folding** | `python/baseline.py` (the canonical protocol), ~20 follow-up experiments, LigandMPNN scoring and integration |
| `dev` | **Boltz-2 embedding generation** | scripts that fold the 75 HLA α1/α2 domains and pool the trunk representation; the resulting parquets |
| `foundation-models-testing` | **embedding ablation** | 330-run ablation harness, ESM-2 cache, split audit, self-contained HTML report |
| `confidence` | **per-complex co-folding** | Modal jobs that co-fold 1,430 complexes, confidence-feature extraction, predictors; plus the two-route slide deck |
| `inverse-folding-slides` | **inverse-folding write-up** | the IF experiment's code, audit, results and deck |

On `main`: **`presentation.html`** — the stitched 15-slide track talk, and the best single starting
point. Alongside it, the pieces it was assembled from: `index.html` (inverse-folding deck),
`slides.pptx` / `slides (1).pptx` (split and nearest-neighbour), `SerajSLides.pptx`,
`baseline_slides.pdf`, `IFprompt.md`, `allele_data.json`.

---

## Workstream 1 — the sequence baseline (`baseline`)

Reproduces the NetMHCstabpan ANN from Rasmussen et al.: BLOSUM50/5 over the 9-mer and the 34-aa
pseudosequence, a small MLP, MSE on `s`.

Beyond reproduction, this branch ran ~20 probes on the baseline itself — target transform, reduced
alphabets, PCA of BLOSUM, masked positions, PSSM and bilinear heads, ranking losses, overfitting
curves, nearest-neighbour controls. The ones that mattered:

| finding | numbers |
|---|---|
| **Training longer is the cheapest win.** The published 25-epoch protocol is undertrained. | mean per-allele PCC +0.093 / +0.078 / +0.066 across three splits, all significant |
| **A better head beats a better representation.** `final_recipe` arms, 10,000-sample bootstrap over 16 held-out alleles. | baseline 0.422 → bilinear-rank **0.513** (+0.092, CI 0.033–0.148, p=0.001); MLP-raw 0.507; PSSM-rank 0.476 |
| **A nearest-neighbour lookup is a strong, unglamorous competitor.** | peptide-only null 0.295; NN joint allele+peptide **0.440** vs baseline 0.396 — 93% of test rows have an exact peptide match elsewhere in the data |

That last row is a warning, not a result: the dataset's peptide panels are reused across alleles,
so "find the same peptide on a similar allele" is a lot of what a model can learn here.

## Workstream 2 — Boltz-2 groove embeddings (`dev` → `foundation-models-testing`)

**Idea.** Fold the HLA α1/α2 domain (182 aa) with Boltz-2, pool the trunk representation `s`
(384-d per residue) over the **B pocket** (P2 anchor) and **F pocket** (C-terminal anchor) into
768 dims, and swap that in for the 680-dim BLOSUM pseudosequence block.

**Scale.** 330 training runs across 3 splits × 5 seeds, plus ESM-2 (35M and 150M) on both the
peptide and HLA side, an **allele one-hot control**, and a measured **noise floor** (SD 0.0029
global PCC). Zero failures.

**Result under the canonical supertype split** (`python/run_supertype.py`, 40 runs):

| arm | dims | mean per-allele SCC | Δ vs baseline |
|---|---|---|---|
| **BLOSUM-pep \| BLOSUM-HLA (baseline)** | 860 | **0.4626 ± 0.0379** | — |
| ESM2-pep \| Boltz-HLA | 6528 | 0.4206 ± 0.0424 | −0.042 (t=−3.2) |
| BLOSUM-pep \| BLOSUM+Boltz (augment) | 1628 | 0.4183 ± 0.0271 | −0.044 (t=−5.0) |
| BLOSUM-pep \| Boltz-HLA (replace) | 948 | 0.3974 ± 0.0282 | −0.065 (t=−14.8) |
| BLOSUM-pep \| ESM2-HLA | 820 | 0.3151 ± 0.0436 | −0.147 (t=−12.1) |
| BLOSUM-pep \| onehot-HLA (control) | 255 | 0.3148 ± 0.0236 | −0.148 (t=−8.2) |
| *peptide-only null* | 0 | *0.3045 ± 0.0163* | −0.158 |

**Every foundation-model arm loses, every one outside the noise floor.** Mean-pooled ESM-2 on the
HLA side (0.3151) is statistically indistinguishable from an allele one-hot (0.3148).

Three things this workstream established that generalise beyond it:

- **A short training budget manufactures false positives.** At the published 25 epochs, Boltz
  pockets looked like a **+0.032 gain (t=+4.1)**. At 100 epochs the sign flipped to −0.015. The
  dense z-scored block is simply an easier optimisation target; given enough epochs BLOSUM
  overtakes it. *This would have been published as a positive result by any study that fixed
  epochs at the published value and did not check.*
- **Global correlation is inflated.** An HLA-only arm that never sees the peptide still reaches
  0.543 global PCC, because global correlation is dominated by between-allele differences in mean
  stability. Per-allele Spearman is the honest metric.
- **Dimensionality, not representation, was most of ESM-2's peptide-side penalty** — compressing to
  BLOSUM width recovers +0.080. Mean-pooling a PLM over a 9-mer destroys the anchor signal, and the
  larger ESM-2 was *worse* than the smaller one.

📄 **`outputs/report/index.html`** on `foundation-models-testing` is a single self-contained file
(3 MB) — all 330 runs embedded, hand-built SVG charts, six interactive 3D pMHC structures. Opens by
double-click, no server, no network.

## Workstream 3 — per-complex co-folding and model confidence (`confidence`)

The embeddings above are **per-allele**: 75 distinct vectors for 28,166 rows, carrying zero peptide
information by construction. This workstream ran the follow-up that limitation implies.

**Idea.** Co-fold each peptide–HLA pair individually as the native trimer (α1+α2+α3 + β2m + 9-mer,
382 tokens) and read out **Boltz-2's own confidence** in where it placed the peptide — pLDDT, PAE,
ipTM, 39 scalars. The hypothesis: a peptide pinned by its anchors has one feasible pose, so the
model is certain; a loose peptide has many, so confidence drops. Confidence as a readout of
geometric constraint, and constraint should track k_off.

*(Boltz-2's affinity head was deliberately not used: it requires a `ligand` chain, and K_d is
thermodynamic while t½ is kinetic — they coincide only if k_on is constant, which for pMHC it is not.)*

**Scale.** 1,430 complexes, all 67 evaluable alleles, leave-one-allele-out. 12 Modal A100 jobs,
~$9.60 total.

| predictor | mean per-allele SCC (67 alleles) | SE |
|---|---|---|
| trained GBM, 39 features | **0.112** | 0.026 |
| zero-shot rank, `pep_plddt_min` | 0.095 | 0.025 |
| trained StabilityNet, 39 features | 0.091 | 0.025 |
| *reference: peptide-only null* | *0.3045* | |
| *reference: BLOSUM baseline* | *0.4626* | |

**The effect is real and far too small.** t = +4.31 vs zero (p = 5.5e-05), positive on 48 of 67
alleles — about **a third** of what ignoring the HLA entirely achieves. Absolute half-life is not
predictable at all: best MAE 3.07 h against 3.10 h for "predict the training-allele median", and
RMSE 53% *worse* than that trivial baseline.

Two sub-results worth keeping:

- **Fold the biologically correct construct.** Adding α3 and β2m did not change confidence *levels*
  (peptide pLDDT 0.466 → 0.460) but made 25 of 33 features more informative (Wilcoxon p = 0.0005).
  Same uncertainty, better aimed. It cost only 2.5× the runtime for 2× the tokens.
- **Few-allele evaluation on this dataset inflates by ~3×.** The same **pre-registered** feature read
  **0.289** on its 6 discovery alleles, **0.178** on 14, and **0.095** on all 67. Pre-registering it
  in the scoring script *before* folding new complexes is the only reason that is an honest number.

## Workstream 4 — inverse folding (`baseline`, `inverse-folding-slides`, deck on `main`)

**Idea.** Score the peptide with an inverse-folding model (**LigandMPNN**, pinned commit + checkpoint
hash) inside the real HLA pocket: given the 3D backbone, how likely is *this* 9-mer at those
positions? Five PDB templates × 3 decoding seeds = 15 ensemble members, backbone only. Peptides are
threaded onto shared templates, so geometry is shared across alleles. The 50 resulting features
(mean log-prob, 9 per-position log-probs, 40 anchor softmax probabilities at P2/P9) go in frozen on
top of the sequence baseline. Two scoring modes — **independent** (all peptide identities hidden,
~45 s) and **autoregressive** (earlier peptide residues revealed, ~745 s).

**Result** (leave-one-allele-out, 75 folds, MLP 64→32, fixed 20-epoch budget, all 28,166 rows;
the figure after the slash is the nonzero-only sensitivity analysis):

| arm | mean per-allele Spearman | Δ vs matched baseline |
|---|---|---|
| zero-shot IF, independent | 0.130 / 0.122 | — |
| zero-shot IF, autoregressive | 0.134 / 0.117 | — |
| **baseline** (BLOSUM + pseudosequence), MSE | **0.423** / 0.409 | — |
| baseline + IF independent, MSE | 0.476 / 0.451 | **+0.053** [0.021, 0.084], 57/75 alleles improved |
| baseline + IF autoregressive, MSE | 0.493 / 0.453 | **+0.070** [0.045, 0.097], 59/75 alleles improved |
| baseline, **ranking loss** | 0.475 / 0.446 | +0.052 [0.030, 0.074] vs MSE baseline |
| baseline + IF independent, ranking loss | 0.509 / 0.475 | — |

Roughly a third more of the ranking explained, and the gains are largest exactly where the baseline
is weakest (the weaker half of alleles gains 2–3× more than the stronger half).

**Three honest qualifications the authors put in writing:**

- **Autoregressive scoring is not worth its cost.** Its MSE advantage over independent features
  (+0.017) has a CI spanning zero, shrinks once zeros are excluded, and *reverses* under ranking
  loss (−0.003) — for 16× the compute. The simpler independent features are the recommendation.
- **Changing the loss helps about as much as the structure model does.** Swapping MSE for a
  within-allele pairwise ranking loss moves the plain baseline 0.423 → 0.475, comparable to
  everything the inverse-folding features buy.
- **A trivial lookup gets close.** A pseudosequence + peptide Hamming nearest-neighbour with no
  training at all reaches 0.423–0.427 — the trained baseline barely beats memorisation-by-similarity.
  The IF arm still clears it (pseudosequence + IF: **0.470**).

**Controls that were run:**

- **Scramble the HLA sequence inside the same structure** and the zero-shot score collapses,
  0.130 → **0.035**. The signal depends on the real pocket sequence in its structural context.
- **Assembly control, decided label-free before looking at targets**: truncated α1/α2 vs full
  HLA+β2m differ by mean |Δ| 0.016 log units (correlation 0.998), so the cheaper truncated form was
  selected under a pre-declared rule.
- **Decoder validation**: the cached factored decoder was shown equivalent to official LigandMPNN
  scoring, with an explicit causality test that no residue reads its own or a future identity.
- **Native vs shared template**: using an allele's own crystal structure helps in 4 of 5 cases
  (mean 0.139 vs 0.061) — better, but not a universal win.
- The **motivating hypothesis was refuted in-repo**: the idea that the pseudosequence discards a
  conserved "clamp" is false — 6 of the 7 clamp positions are already in it.

**Caveats:** single training seed, fixed budget, descriptive (uncorrected, non-independent)
bootstrap intervals, and the structure model *sees more of the HLA* than the baseline (182 residues
vs 34), so the gain is not yet isolated to geometry alone.

---

## Cross-cutting lessons

These cost real runs to learn and are the most transferable part of the repo.

1. **Fix the training budget and you will manufacture a positive result.** A feature that merely
   helps a model converge faster is indistinguishable from a feature that adds information, unless
   you train both arms to convergence. One workstream's +0.032 "gain" became −0.015 at 4× epochs.
2. **Measure your noise floor before claiming a delta.** The published script seeds NumPy but not
   torch, so "seed 42" runs differ by ~0.005 global PCC. Nothing smaller than that is a result.
3. **Pair comparisons within a seed on allele-holdout splits.** Baseline seed SD on a naive
   leave-allele-out split was **0.081 — 61× the random split**. Which alleles get held out matters
   more than the model does. A single-seed leave-allele-out number is close to meaningless.
4. **Few-allele evaluation inflates by about 3×** on this dataset, demonstrated twice with two
   different features.
5. **Pre-register the feature.** Best-of-39 on six alleles read 0.289; the same feature read 0.095
   on 67. Writing it into the scoring script as a constant *before* collecting new data is the only
   reason the shrinkage was visible rather than hidden by a second scan.
6. **Global correlation rewards modelling the allele marginal**, not the peptide–HLA interaction.
   Report per-allele.
7. **Split on pseudosequence, not allele name**, and require a minimum measurement count per
   held-out allele. Both lessons were learned the expensive way.
8. **Never move binary files through a PowerShell text redirect.** The handed-over embedding
   parquets arrived irreversibly corrupted (BOM prepended, every non-decodable byte collapsed to
   U+FFFD, files inflated 1.7×) and had to be re-extracted with `git cat-file` through a byte-safe
   pipe.
9. **Open the deliverable the way a user will.** The HTML report's 3D panels were blank on a plain
   double-click; the original check had passed only because headless Chrome ran with
   `--allow-file-access-from-files`, a permission a real double-click does not have.

## What was not tested

- **Fine-tuning any foundation model.** Nothing here is anything but frozen features — and
  fine-tuning is where PLMs usually earn their keep. No CUDA was available (Intel Arc iGPU).
- **Per-complex trunk embeddings.** The co-folding pass wrote them (`embeddings_*.npz`) and they are
  untouched. Strictly more information than the confidence scalars that were read instead.
- **Leave-one-supertype-out**, the harder generalisation test. The gap between it and
  leave-one-allele-out would quantify how much performance depends on having relatives in training.
- **The confidence-augmentation arm at full scale** — needs the full 28,166-row co-fold cache,
  ~$52 of GPU at the measured 8.9 s/complex.
- **Proper leave-one-allele-out CV over all 75 alleles for the embedding arms**, which is what would
  actually settle the unseen-allele case.

---

## Running it

Python 3.12, [`uv`](https://docs.astral.sh/uv/) for dependencies. Every workstream lives on its own
branch; check one out and work there.

```bash
git clone git@github.com:efe-a-g/HLAxPeptide-Protein-Engineering.git
cd HLAxPeptide-Protein-Engineering
git checkout baseline          # or: foundation-models-testing | confidence | dev
uv sync
```

**Baseline and its follow-ups** (`baseline`):

```bash
uv run python python/baseline.py        # canonical split + metric; import these elsewhere
uv run python python/final_recipe.py    # the bilinear-rank / PSSM-rank heads
uv run python python/mpnn_score.py      # LigandMPNN scoring
uv run python python/mpnn_integrate.py  # IF features on top of the baseline
```

**Embedding ablation** (`foundation-models-testing`), ~2 h on CPU end to end:

```bash
bash scripts/extract_embeddings.sh      # binary-safe; verifies PAR1 head and tail
uv run python python/verify_data.py     # dataset + embedding facts
uv run python python/verify_splits.py   # leakage audit
uv run python python/embed_esm.py       # ESM-2 cache
uv run python python/run_grid.py        # phase 1: 135 runs, ~26 min
uv run python python/run_phase3.py      # 150 runs, ~70 min
uv run python python/run_supertype.py   # the canonical-protocol re-run
uv run python python/aggregate_supertype.py
uv run python python/build_report.py    # → outputs/report/index.html
```

**Co-folding** (`confidence`) — needs a [Modal](https://modal.com) account and a GPU budget:

```bash
modal run confidence/python/fold_trimer.py    # ~8.9 s/complex on A100
uv run python confidence/python/analyze_stage1.py
uv run python confidence/python/predict_halflife.py
```

**Boltz-2 HLA embeddings** (`dev`):

```bash
modal run scripts/modal_boltz_embeddings.py   # 75 alleles
uv run scripts/pocket_pool_embeddings.py      # B/F pocket pooling
```

**Inverse folding** (`inverse-folding-slides`) — this workstream uses its own `.venv` and
`requirements.lock.txt` rather than `uv`, and needs LigandMPNN vendored at the pinned commit plus
the five PDB templates (restore commands are in `inverse-folding/README.md`):

```bash
cd inverse-folding
python -m venv .venv && .venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/splits.py
.venv/bin/python scripts/structural_pipeline.py --stage full --assembly truncated --mode both
.venv/bin/python scripts/baselines.py --stage full
.venv/bin/python scripts/aggregate_results.py && .venv/bin/python scripts/write_findings.py
.venv/bin/python scripts/build_report.py      # → inverse-folding/report.html
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

`outputs/artifact_manifest.json` records a SHA-256 for every deliverable and asserts all 21
required (model, loss) runs completed.

## Where to look first

| you want | open |
|---|---|
| **the whole story in 15 slides** | **`presentation.html` on `main`** |
| the full embedding ablation, interactively | `outputs/report/index.html` on `foundation-models-testing` |
| the reasoning, decision by decision | `outputs/LOG.md` on `foundation-models-testing` (entries 0–10) and `confidence/LOG_entry_11.md` |
| the two-route talk | `slides/foundation_model_routes_deck.html` + `slides/SPEAKER_NOTES.md` on `confidence` |
| the inverse-folding write-up | `inverse-folding/report.html` on `inverse-folding-slides` |
| the inverse-folding talk | `index.html` on `main` |
| the canonical protocol, in code | `python/baseline.py` on `baseline` |

## Credits

Team: Efe Ali Görgüner, Karam Al-Robaie, Arya Saranathan, David Theodor Nimrichtr, S. Ali.
Challenge set by Serova. Dataset from Rasmussen et al. (2016). Supertype assignments from
Sidney et al. (2008). Structures from the RCSB PDB (usage policy recorded in `.licenses/`).
Parts of the analysis and tooling were written with Claude Code.
