## Entry 11 — Per-complex co-folding: the structure model's own confidence

Entries 0-10 tested **frozen per-allele embeddings**. Their central limitation was
stated in Entry 0 and never addressed: the Boltz-2 embeddings are 75 distinct
vectors for 28,166 rows, so they carry *zero* peptide information by construction.
Entry 7's limits section named the fix — "a **per-complex** embedding (one structure
per peptide-HLA pair) is a genuinely different and untested experiment, and is the
single most promising follow-up".

This entry runs that experiment, with a different readout than embeddings.

**Question.** Co-fold each peptide-HLA complex individually and read Boltz-2's own
*confidence* in where it placed the peptide. Does that confidence predict half-life?

**Why confidence rather than the affinity head.** Boltz-2's affinity module requires
the binder to be a `ligand` chain; there is no protein-protein affinity path, so a
peptide cannot be scored by it. A 9-mer *could* be declared as a SMILES ligand (all
5,633 unique dataset peptides are 48-100 heavy atoms, none near the 128-atom cap),
but that forfeits the MSA, the peptide-bond prior and protein atom typing, and feeds
a head trained on drug-like molecules. Separately, affinity is the wrong physical
quantity: Kd = k_off/k_on is thermodynamic, t-half = ln2/k_off is kinetic, and they
coincide only if k_on is constant — which for pMHC it is not, which is why
NetMHCstabpan exists apart from NetMHCpan and why this dataset was generated at all.
Boltz's `affinity_pred_value` is log10(IC50 in uM), a competition-assay quantity
further still from k_off.

So: co-fold as protein chains, read pLDDT / PAE / ipTM. In-distribution (pMHC-I is
abundant in the PDB), genuinely zero-shot, and free from the same pass that yields
per-complex trunk embeddings.

**Hypothesis.** A peptide pinned by its anchors has one feasible position, so the
model is certain; a loosely-fitting peptide has many, so confidence drops. Confidence
is then a readout of geometric constraint, and constraint should track k_off. This is
a leap, not a guarantee.

Machine: Boltz-2 2.2.1 on Modal, 1x A100-40GB per container. 12 jobs, ~$9.60 total.

---

### Stage 1 — alpha1/alpha2 + peptide, 191 tokens (300 complexes, 10 alleles x 30 peptides)

Deliberately 10 alleles x 30 peptides rather than 300 alleles-worth of singletons:
the first question is whether confidence varies *across peptides within an allele*,
and 4 peptides per allele cannot measure that.

**Variance check: passed.** 0 of 33 confidence features are per-allele constants;
within/between-allele SD ratios run 1.0-4.9. The degenerate failure mode — Boltz
uniformly confident about every peptide in a groove, so the features encode only
allele identity — did not occur.

| predictor | mean per-allele PCC |
|---|---|
| best single feature (`pocketB_plddt_mean`) | 0.148 +/- 0.214 (SE 0.068) |
| leave-one-allele-out GBM, 34 features | 0.138 +/- 0.125 |
| leave-one-allele-out ridge, 34 features | 0.019 |

Against the peptide-only null of 0.3045 (canonical protocol), that is 49%. Measured
3.4 s/complex on GPU; 300 complexes in 1965 s for ~$1.15.

### Entry 5's lesson, re-learned: beta2m is not optional, but not for the expected reason

Absolute confidence in stage 1 was low (`hla_plddt_mean` ~0.57, `pep_plddt_mean`
~0.46). Hypothesis: the alpha1/alpha2-only construct is a less stable fold, so Boltz
is unsure about everything and the general fog drowns the peptide-specific part.

Control: 150 complexes (6 alleles x 25 peptides, the *same* peptides) re-folded as
the native trimer — full alpha ectodomain (alpha1+alpha2+alpha3, 274 aa) + beta-2-
microglobulin (99 aa) + peptide = 382 tokens. The alpha3 domain is 97.8% identical
between the HLA-A and HLA-B references, so one locus reference was appended to each
allele's own alpha1/alpha2 rather than fetching 75 full-length sequences; HLA-A*03:01's
alpha1/alpha2 matches the HLA-A reference at 100%, which pins the numbering exactly.

**The hypothesis was wrong.** Confidence levels did not move:

| | alpha1/alpha2 only | full trimer | delta |
|---|---|---|---|
| `hla_plddt_mean` | 0.572 | 0.567 | -0.004 |
| `pep_plddt_mean` | 0.466 | 0.460 | -0.006 |
| `conf_iptm` | 0.710 | 0.691 | -0.019 |

**But the confidence became more informative.** 25 of 33 features improved in
|per-allele SCC| (paired Wilcoxon p=0.0005); median |SCC| rose 0.058 -> 0.104. The
best feature, `pep_plddt_min`, went -0.040 -> +0.289 (SE 0.061), positive on all six
alleles, paired t=+3.81 (p=0.012). Same uncertainty, better aimed: with the correct
scaffold the model stops hedging about the fold and its residual doubt is about the
peptide.

Cost of the trimer: 8.6 s/complex, only 2.5x the 191-token run for 2x the tokens —
better than the N^2-N^3 triangle-op scaling predicts. Biologically correct and
affordable are not in tension here.

### Entry 6's lesson, re-learned: the reported effect was selection, and it was caught by pre-registration

`pep_plddt_min` was chosen as best-of-33 on six alleles — exactly the move that
produced Entry 1's withdrawn claim. Before folding anything further, the feature was
**written into `predict_halflife.py` as a constant** with a comment forbidding
re-selection on new data.

200 further complexes, 8 alleles x 25 peptides: 4 alleles paired against the stage-1
peptides, and **4 alleles never folded before**.

| set | mean per-allele SCC, `pep_plddt_min` |
|---|---|
| discovery (the 6 alleles it was chosen on) | **+0.289** (SE 0.061) |
| **4 never-folded alleles** | **+0.090** (SE 0.176) |
| 4 previously-folded alleles | +0.100 (SE 0.098) |

**67% shrinkage.** Had the 39 features been re-scanned on the new batch, something
would have scored ~0.3 and been reported as a replication. The pre-registration is
the only reason this is an honest number.

---

### Final — 1,430 complexes, all 68 evaluable alleles, leave-one-allele-out

Nine parallel containers (Modal's concurrency ceiling is 10), sharded **by allele** so
each container fetches only its own alpha-chain MSAs, rows **interleaved** within a
shard so the 28-minute cap costs peptides evenly rather than whole alleles. Seven of
nine hit the cap at 120/144; partial outputs harvested as designed. 20-25 peptides per
allele.

Leave-one-allele-out over every allele — the CV scheme Entry 4 named as the fix for
that split's pathological seed variance. It is deterministic: no seed, every allele
tested exactly once.

| predictor | 6 alleles | 14 alleles | **67 alleles** | SE |
|---|---|---|---|---|
| trained GBM, 39 features | 0.241 | 0.112 | **0.112** | 0.026 |
| zero-shot rank, `pep_plddt_min` | 0.289 | 0.178 | 0.095 | 0.025 |
| trained StabilityNet, 39 features | 0.229 | 0.162 | 0.091 | 0.025 |
| zero-shot rank, `pep_plddt_mean` | 0.217 | 0.192 | 0.082 | 0.028 |
| calibrated zero-shot, `pep_plddt_min` | 0.244 | 0.175 | 0.081 | 0.025 |
| calibrated zero-shot, `pep_plddt_mean` | 0.223 | 0.205 | 0.077 | 0.028 |

Reference: peptide-only null **0.3045**, BLOSUM baseline **0.4626**.

**The effect is real and far too small.** GBM's 0.112 is t=+4.31 vs zero
(p=5.5e-05), positive on 48 of 67 alleles. Boltz confidence does carry
peptide-specific stability information — roughly **a third** of what ignoring the HLA
entirely achieves.

Two readings of the convergence worth separating. The **single-feature** arms kept
falling as alleles were added, because their early values were selection-inflated.
The **trained** arms stopped falling (GBM 0.112 -> 0.112) once each fold had 67
alleles instead of 13, because they had been data-starved. Two different failure
modes that are indistinguishable at six alleles.

### Hours are not predictable, only order

| | MAE | RMSE |
|---|---|---|
| **trivial: predict the training-allele median** | **3.10 h** | **5.04 h** |
| best model | 3.07 h | 7.72 h |

MAE ties the trivial baseline and RMSE is 53% worse. Reporting MAE alone would have
looked like a working model. Per-allele SCC ~0.11 alongside global SCC ~0.25 and
AUC@1h ~0.63 reproduces the exact signature Entry 3 recorded for the embeddings on
leave-allele-out: *the absolute stability scale for an unseen allele is lost while
within-allele ranking partially survives.* Same property, different features — so it
is a property of the problem.

### Bookkeeping

- `HLA-B*39:06(C67S)` cannot be scored: all 20 sampled peptides share one measured
  half-life, so Spearman is undefined. Hence 67 alleles, not 68.
- `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share a pseudosequence, so leave-one-
  **allele**-out leaks between them (`split_by_supertype` groups by pseudosequence and
  does not). Their scores are -0.017 and +0.059; excluding both moves GBM from +0.1119
  to +0.1147, i.e. the leak made the result marginally pessimistic. Not re-run.
- Leave-one-allele-out is an *easier* test than the canonical supertype-stratified
  split, because an allele's supertype relatives stay in training. These numbers are
  therefore optimistic relative to the repo's 0.4626 and are not interchangeable with
  it. Leave-one-supertype-out was not run.

---

## Answer

> *Are open protein foundation models useful for predicting peptide-HLA class I stability?*

Entries 0-10 answered **no** for frozen per-allele embeddings. This entry extends that
to the untested case those entries flagged as most promising:

**Nor does per-complex co-folding help, read through the model's own confidence.**
1,430 complexes individually co-folded as native trimers across 67 alleles,
leave-one-allele-out, six predictor variants: best mean per-allele Spearman
**0.112 +/- 0.026** against a peptide-only null of **0.3045**. The signal is real,
reproducible, and a third of what ignoring the HLA achieves. Absolute half-life is not
predictable at all.

### The methodological finding, which may outlast the negative result

**Few-allele evaluation on this dataset inflates by ~3x**, demonstrated twice with two
different features: `pocketB_plddt_mean` read 0.148 on 10 alleles and did not survive a
change of geometry; `pep_plddt_min` read 0.289 on 6 alleles and 0.095 on 67. This is the
quantitative form of the power limitation Entry 4 flagged (seed SD 0.081, 61x the random
split). Any single-split or few-allele number on this problem should be assumed inflated
by about that factor until shown otherwise.

### What was not tested

1. **Per-complex trunk embeddings.** The co-folds wrote `s` slices for the peptide
   residues and both pockets (`embeddings_*.npz`), and they are untouched. That is the
   original Entry 7 follow-up; this entry tested confidence instead, which is cheaper to
   read but strictly less information.
2. **Fine-tuning.** Still nothing. Still the place PLMs usually earn their keep.
3. **Leave-one-supertype-out**, the harder generalisation test. The gap between it and
   leave-one-allele-out would quantify how much performance depends on having relatives
   in training — which nothing in this repo currently measures.
4. **The augmentation arm at full scale.** BLOSUM-860 + ~39 confidence scalars = 899
   dims, a 4.5% width increase (against the 89% of Entry 7's failed 1,628-dim arm).
   `stage_b.py` runs it, but on 1,430 rows the BLOSUM reference arm itself only reaches
   0.060 against its true 0.4626, so the comparison is not yet measurable. It needs the
   full 28,166-row cache, ~$52 of trimer folding at the measured 8.9 s/complex.
