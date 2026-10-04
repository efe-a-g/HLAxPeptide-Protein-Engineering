# Peptide–HLA stability: reproducible empirical evaluation

Open **[report.html](report.html)** for the offline research report, with embedded figures. **[results.csv](results.csv)** contains the full metric table, including unavailable cells and coverage. These are local research artifacts; no website or service is required.

The experiment asks whether LigandMPNN structural features improve a supervised BLOSUM-peptide plus HLA-pseudosequence baseline. The primary endpoint is mean within-allele Spearman under leave-one-allele-out (LOAO) evaluation. Every metric is reported for all rows and positive-half-life rows separately. A negative result is a valid result.

## Important findings and interpretation

- The input contains 28,166 unique allele–peptide pairs, 5,633 peptides, and 75 alleles; 5,679 targets (20.16%) are exactly zero. Matching the original paper's measured-table size does **not** establish that zeros are synthetic padding. Their exact convention and assay detection threshold remain unknown.
- No raw replicate measurements occur in this export. A numerical replicate noise ceiling is **not estimable**. Cross-allele measurements of a peptide are different complexes, not replicate measurements. Correlation figures explicitly state that the ceiling is unavailable.
- Joint pseudosequence recovery has four valid ordered index sets. The externally annotated canonical set matches every row but is not uniquely identifiable from this table. Six of the seven proposed clamp positions already occur in the pseudosequence, so the experiment cannot be interpreted as restoring a wholly omitted conserved clamp.
- Allele counts range from 7 to 1,070. LOAO includes closely related alleles and peptides observed with other alleles; it is not a joint allele-and-peptide novelty test. The random split is a leakage diagnostic, never the headline result.
- Full-data empirical motifs are descriptive artifacts only. Supervised PWMs, target means, and structural feature scaling are fitted separately using each training fold.

See the report for actual scores and [notes/scientific_references.json](notes/scientific_references.json) for primary references and scientific caveats.

## Environment and immutable inputs

Commands below run from the repository root. The completed local environment uses Python **3.14.6**, a project-local `.venv`, and the versions in [requirements.lock.txt](requirements.lock.txt). CPU execution is supported; a GPU is not required. Package availability may differ by Python version or platform.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

The supplied table is `resources/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv`. Its SHA-256 is:

```text
17c574c2c6d2746a238804bd73fa05279ae23e476be5df124dc084e116635658
```

The official [LigandMPNN repository](https://github.com/dauparas/LigandMPNN) is vendored at `vendor/LigandMPNN`. The recorded source commit is `26ec57ac976ade5379920dbd43c7f97a91cf82de`. The checkpoint is `ligandmpnn_v_32_010_25.pt`, SHA-256:

```text
161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970
```

To restore missing model inputs in a fresh checkout:

```sh
git clone https://github.com/dauparas/LigandMPNN.git vendor/LigandMPNN
git -C vendor/LigandMPNN checkout 26ec57ac976ade5379920dbd43c7f97a91cf82de
mkdir -p vendor/LigandMPNN/model_params outputs/structure/pdb
curl -fL https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt \
  -o vendor/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
for pdb in 1HHK 5IB1 7MLE 4NQX 3LKN; do
  curl -fL "https://files.rcsb.org/download/${pdb}.pdb" -o "outputs/structure/pdb/${pdb}.pdb"
done
```

The five PDB templates span A*02:01, B*27:05, A*03:01, A*01:01, and B*35:01. Template metadata record actual resolution, chain identities, native peptide, sequence agreement, spatial adjacency, missing backbone atoms, and hashes. Saved backbone-only templates preserve distinct HLA, peptide, and, for full assemblies, β2-microglobulin chains.

## Reproduce the audit and fixed splits

```sh
.venv/bin/python scripts/audit_data.py
MPLCONFIGDIR=/tmp/peptide-matplotlib .venv/bin/python scripts/audit_figures.py
.venv/bin/python scripts/splits.py
```

`audit_data.py` also runs with the standard library alone. Its defaults are numerical offset ε = 0.05 h, stable-peptide threshold > 2 h, and motif pseudocount 0.5 per amino acid per position. The offset is not an empirically established assay floor.

Seed `20261003` fixes a label-independent allele-balanced 1,000-row pilot, 80/20 random holdout, 80/20 peptide-group holdout, and all 75 LOAO folds. A split manifest saves test row IDs and the dataset universe; training membership is their set difference. The pilot uses original CSV row IDs.

## Reproduce supervised baselines

Run and inspect the pilot before the full dataset:

```sh
.venv/bin/python scripts/baselines.py --stage pilot
.venv/bin/python scripts/baselines.py --stage full
```

Each output directory saves its full `config.json`, compressed row-level predictions, timing records, `results.csv`, and `per_allele_metrics.csv`. The fixed default budget is 20 epochs, batch size 1,024, Adam learning rate 0.001, and hidden widths 64/32, with two CPU threads. Fold and representation seeds derive deterministically from seed `20261003`.

The baseline ladder includes train-allele log means, train-fold PWMs, a 32-dimensional learned categorical allele embedding plus BLOSUM50 peptide encoding, and BLOSUM50 peptide plus one-hot pseudosequence. LOAO categorical embeddings and allele PWMs remain unavailable. The allele-mean baseline uses a global training-mean fallback for unseen alleles. The learned heads compare MSE, a one-sided squared censoring surrogate, and within-allele pairwise ranking. The surrogate assumes c = 0.1 h and uses threshold log(c + ε); it is a sensitivity analysis, not a verified censoring model or Tobit likelihood.

`--resume` reads the saved row-level predictions to avoid repeating completed fold/representation/loss cells. Reuse an output directory only with its original configuration; select a new `--output` path for a changed budget, representation, loss, data universe, or feature archive.

## Reproduce structural scoring

```sh
.venv/bin/python scripts/structural_pipeline.py --stage validate
.venv/bin/python scripts/structural_pipeline.py --stage controls
```

Validation compares the factored decoder to official LigandMPNN scoring and verifies that peptide-independent predictions ignore peptide identities while responding to HLA identities. The 500-row allele-balanced assembly control compares alpha1/alpha2 truncation with the full HLA/β2-microglobulin assembly before full scoring. Consult `outputs/structure/assembly_selection.json` for the saved decision and criterion.

The label-free control selected the truncated assembly: mean absolute ensemble-score difference was 0.01614 log units, below the predeclared 0.02 criterion; full/truncated score correlation was 0.9983. This is evidence of score stability, not proof of superior predictive accuracy. Reproduce the selected runs with:

```sh
.venv/bin/python scripts/structural_pipeline.py --stage pilot --assembly truncated --mode both
.venv/bin/python scripts/structural_pipeline.py --stage full --assembly truncated --mode both
.venv/bin/python scripts/structural_pipeline.py --stage conditional_controls --assembly truncated
.venv/bin/python scripts/structural_analysis.py
```

The additional `conditional_controls` stage evaluates shuffled HLA and full-assembly sensitivity for autoregressive scoring on 500 allele-balanced rows. It is an exploratory check after the main assembly choice was fixed, not a second selection procedure. Results are saved in `outputs/structure/conditional500_controls.json` and matching feature archives.

Three deterministic decoding seeds, `17`, `29`, and `43`, are used for each of five templates. “Unconditional” means **all peptide identities hidden while fixed HLA identities remain visible**. It is not the official geometry-only `use_sequence=False` path. Conditional scoring makes HLA context precede a randomized autoregressive peptide order. Fixed-context decoder states are cached; an independent causality test verifies that a residue cannot read its own or a future peptide identity.

Scores average log likelihood across template/order members. `mean_logp` is the mean over nine observed-residue log probabilities; summing instead rescales it by nine without changing ranking. Saved features include per-position log probabilities, the two 20-way anchor distributions, and ensemble score spread. This spread describes template/order variation; it is not an experimental uncertainty interval.

The model receives backbone geometry and fixed sequence context. Atomic side-chain context and allele-specific side-chain repacking are not used. Geometry is shared across threaded alleles. Scrambled-HLA controls test whether scores respond to sequence context; they do not by themselves prove calibrated allele specificity or kinetic prediction.

Caches in `outputs/structure/cache` include the checkpoint hash, upstream commit, scoring-algorithm hash, template hash, allele/HLA sequence, mode, decoding seed, and scrambling flag. Conditional archives index peptide contents internally, so larger datasets reuse previously scored peptides and preserve requested row order. Each cache manifest preserves the context key. Check `outputs/structure/provenance.json`, `templates.json`, and run JSON files for exact executed provenance. Changes to the scoring methods, template builder, or upstream model code invalidate cache keys; the complete pipeline file hash is also saved as provenance.

## Fit structural feature ablations

The planned primary representation comparison uses **MSE**, comparing the MSE baseline with MSE baseline-plus-structure heads. The combined fifty-feature variant is the primary augmentation; the individual components are ablations. After feature extraction, these commands train the four added-feature variants without repeating standalone baselines:

```sh
.venv/bin/python scripts/baselines.py --stage full --losses mse --representations '' \
  --structural outputs/structure/features_unconditional.npz \
  --output outputs/baselines/full_unconditional
.venv/bin/python scripts/baselines.py --stage full --losses mse --representations '' \
  --structural outputs/structure/features_conditional.npz \
  --output outputs/baselines/full_conditional
```

Repeat with `--stage pilot`, the corresponding `features_*_pilot.npz` archive, and a separate pilot output directory to verify the integration first. The structural mode is encoded in each representation name. Variants append one mean score, nine per-position scores, forty anchor probabilities, or all fifty features to the supervised baseline. Scaling statistics are fitted only on the training fold. Hidden layers and training budgets remain fixed; the input dimension and therefore the first-layer parameter count change.

## Exploratory structural augmentation with ranking loss

The completed baseline comparison found that ranking loss outperformed MSE: LOAO macro Spearman was 0.475 versus 0.423 on all rows and 0.446 versus 0.409 on positive-only rows. This observation motivated an **exploratory** follow-up using combined structural features with the stronger ranking objective. It does not replace or retroactively redefine the original MSE comparison.

```sh
.venv/bin/python scripts/baselines.py --stage full --losses ranking --representations '' \
  --structural-features all --exploratory \
  --structural outputs/structure/features_unconditional.npz \
  --output outputs/baselines/full_unconditional_ranking
.venv/bin/python scripts/baselines.py --stage full --losses ranking --representations '' \
  --structural-features all --exploratory \
  --structural outputs/structure/features_conditional.npz \
  --output outputs/baselines/full_conditional_ranking
```

The corresponding integration pilots use `--stage pilot`, `features_unconditional_pilot.npz` / `features_conditional_pilot.npz`, and output directories `outputs/baselines/pilot_unconditional_ranking` / `outputs/baselines/pilot_conditional_ranking`. These runs retain all three split types and the original fixed training budget. `--structural-features all` selects the combined fifty-feature variant only; `--exploratory` records the follow-up status in the saved configuration. Compare each augmented ranking head with the **ranking baseline**, while comparing augmented MSE heads with the **MSE baseline**. For a given representation and fold, initialization is identical across loss variants.

## Assemble the report and validate

Once all intended runs are complete:

```sh
.venv/bin/python scripts/aggregate_results.py
.venv/bin/python scripts/write_findings.py
MPLCONFIGDIR=/tmp/peptide-matplotlib .venv/bin/python scripts/build_report.py
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v
.venv/bin/python scripts/finalize_artifacts.py
```

Aggregation excludes incomplete full-data LOAO runs. `outputs/included_runs.json` identifies the inputs actually used. The report embeds its figure images, so it can be read offline; raw results and file links remain local companion artifacts.

The tests check full-data split isolation, deterministic target-independent sampling, average ranks for ties, macro versus size-weighted metrics, nonzero filtering, missing-prediction coverage, autoregressive masking, official-decoder equivalence, cache separation, and feature averaging. They use a small number of structural forward passes rather than repeating whole experiments.

## Output map

| Path | Contents |
| --- | --- |
| `report.html` | Offline research report and embedded visualizations |
| `results.csv` | Long-form representation/split/loss/subset/aggregation/metric results |
| `outputs/per_allele_metrics.csv` | Within-allele results and valid row counts |
| `outputs/paired_comparisons.csv` | Paired allele-level changes and descriptive bootstrap intervals |
| `outputs/audit/` | Audit JSON, allele counts, targets, anchor composition, empirical motifs, all four index mappings, canonical mapping, unavailable noise-ceiling record |
| `outputs/splits/{pilot,full}/` | Seeded split memberships, universe, and leakage checks |
| `outputs/baselines/` | Pilot/full configurations, predictions, timings, and individual metric tables |
| `outputs/structure/` | Decoder validation, template/PDB inputs, provenance, controls, feature archives, cache manifests, structural analysis, and PNG/PDF figures |
| `outputs/figures/` | Audit and supervised figures in PNG and vector PDF |
| `outputs/artifact_manifest.json` | Final completeness checks and SHA-256 hashes of deliverables and run artifacts |
| `notes/scientific_references.json` | Primary references and caveats |

Spearman and log-target Pearson require at least three valid observations and a variable target. Constant predictions are assigned an explicitly operational zero to represent no ranking; mathematically their correlations are undefined. Missing model predictions remain unavailable. AUC requires both threshold classes. Macro metrics weight eligible alleles equally; size-weighted metrics weight them by valid row count. Always inspect eligibility and coverage alongside the value.

The paired bootstrap resamples alleles and is descriptive: related alleles and overlapping training folds are dependent, and a single training seed does not characterize optimization uncertainty. The ranking-augmentation follow-up was motivated by observed baseline evaluation results and is explicitly exploratory; it does not supply an independent confirmatory test. The intervals do not establish significance, universal biological positional independence, or an experimental noise ceiling. Similarity-reduced peptide partitioning and fully independent prospective data remain follow-up work.
