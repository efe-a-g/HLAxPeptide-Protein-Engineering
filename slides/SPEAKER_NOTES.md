# Speaker notes — two foundation-model routes to peptide-HLA stability

Deck: `foundation_model_routes_deck.html` (self-contained reveal.js; arrow keys to advance,
**down arrow** on slide 2 for the two backup tables, `F` for fullscreen, `O` for overview).

Three slides. Route names on the slides are *using peptide shape embedding* and *using confidence
matrix*. If asked: the shape embedding describes the HLA groove, not the peptide — the peptide side
is encoded exactly as the baseline does.

## Slide 1 — the two approaches (merged: cartoons on top, pipeline written beneath)
- Data: Rasmussen et al., 28,166 measurements, 75 alleles, all 9-mers. Target s = 2^(-1 h / t-half).
- Route A (`dev` branch, entries 0-10): Boltz-2 folds the HLA alpha1/alpha2 domain alone (182 aa);
  the trunk representation is pooled over the B- and F-pocket residues into 768 dims and swapped in
  for the 680-dim BLOSUM pseudosequence block of the baseline network.
- Route B (`confidence` branch, entry 11): every peptide-HLA pair co-folded as the native trimer
  (alpha1+alpha2+alpha3 + beta2m + peptide = 382 tokens), 1,430 complexes, and the model's own
  confidence in the pose read out as 39 scalars.
- The graphic's Route B peptide is drawn with one low-confidence residue because that is the
  mechanism being tested: the strongest single feature was the *minimum* per-residue pLDDT.

## Slide 1, the cartoons
- Left: the alpha1/alpha2 groove on its own, which is literally all Route A ever folded (182 aa).
- Right: the native trimer with the 9-mer in the groove, which is what Route B folded 1,430 times.
- Structure is 1M6O (HLA-B*44:02 + 9-mer) from the repo's reference set — an experimental crystal
  structure, not a prediction. Say so if anyone asks.
- The paler peptide residue is decorative, echoing the mechanism: the best single feature was the
  *minimum* per-residue pLDDT, i.e. the worst-placed residue, not the average.

## Slide 2 — against the benchmark
- Metric: mean per-allele Spearman. BLOSUM baseline 0.4626 +/- 0.0379 (5 seeds).
  Peptide-only null 0.3045 — throws the HLA away entirely.
- Route A best arm: ESM-2 peptide + Boltz pockets, 0.4206. Every embedding arm lost, all
  outside the 0.0029 noise floor. Boltz pockets replacing the pseudosequence: 0.3974 (-0.065, t=-14.75).
- Route B best arm: gradient boosting on 39 confidence features, 0.1119 +/- 0.0259;
  t = +4.31 vs zero, p = 5.5e-05, positive on 48 of 67 alleles.
- If asked "is it significant": yes, and that is the point — a real effect roughly a third of the null.
- Caveat to state out loud: Route B was scored leave-one-allele-out, which keeps an allele's
  supertype relatives in training. It is the *easier* split, so 0.11 is the optimistic number and is
  not interchangeable with the 0.4626/0.3045 references.
- Absolute hours (backup slide): best model MAE 3.07 h vs 3.10 h for "predict the training-allele
  median"; RMSE 7.72 h vs 5.04 h, i.e. 53 % worse. MAE alone would have looked like a working model.

## Slide 3 — why, and the keeper
- Route A: the pooled pocket positions overlap heavily with the 34 NetMHCpan pseudosequence
  positions, so the embedding re-describes ground BLOSUM already covers. Frozen and per-allele it is
  75 vectors for 28,166 rows — no peptide information at all. Mean-pooled ESM-2 on the HLA side
  (0.3151) is indistinguishable from an allele one-hot (0.3148).
- Route A's other trap: a +0.032 apparent gain at the published 25-epoch budget reversed sign at
  100 epochs. The largest real gain anywhere in the ablation was training longer.
- Route B half-worked where it was tested: folding the native trimer left confidence levels flat
  (peptide pLDDT 0.466 -> 0.460) but made 25 of 33 features more informative (Wilcoxon p = 0.0005).
  Same uncertainty, better aimed.
- The keeper: few-allele evaluation inflates this metric by roughly 3x. The same pre-registered
  feature read 0.289 on its 6 discovery alleles, 0.178 on 14, 0.095 on all 67. Pre-registering it in
  the scoring script before folding new complexes is the only reason that is an honest number.
- Cost, if asked: the whole co-folding campaign was 12 Modal A100 jobs, ~$9.60.
- Not tested: fine-tuning, per-complex trunk embeddings (the npz files exist, untouched),
  leave-one-supertype-out.
