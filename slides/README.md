# slides — structural foundation models on peptide-HLA stability

`foundation_model_routes_deck.html` is a **self-contained** reveal.js 6.0.2 deck: the reveal
CSS/JS, the theme fonts, 3Dmol.js 2.5.5, the structure coordinates and both figures are inlined as
data URIs, so it opens from `file://` with no server and no network. Arrow keys advance; `down` on
slide 2 reveals the two backup tables; `F` fullscreen, `O` overview.

Three slides:

1. **Approaches with structural foundational models that did not perform better** — two columns,
   each with a spinning 3D cartoon of what that route actually folded, then the route's pipeline
   written out beneath it. Left: *using peptide shape embedding*. Right: *using confidence matrix*.
2. Both routes on one bar chart against the benchmark, with the full arm tables as sub-slides.
3. Why each closed, plus the few-allele inflation result.

A naming note for whoever presents it: the left route's embedding describes the **HLA groove**, not
the peptide — the peptide is encoded exactly as the baseline does. The slide name is the one the
team chose; the step list under it is precise.

## Where the numbers come from

- Shape-embedding arms: `outputs/supertype/results.jsonl` on `foundation-models-testing`
  (canonical supertype leave-allele-out, 5 seeds). BLOSUM baseline 0.4626 +/- 0.0379,
  peptide-only null 0.3045, best arm (ESM-2 peptide + Boltz pockets, 6,528 dims) 0.4206 +/- 0.0424.
  The pure swap, Boltz pockets replacing the pseudosequence, is 0.3974.
- Confidence-matrix predictors: `confidence/outputs/predictAll_predictor_comparison.csv`
  (leave-one-allele-out, 67 alleles, 1,430 co-folded trimers). Best arm 0.1119 +/- 0.0259.
- Stated on the slide, not buried: leave-one-allele-out is the *easier* split, so 0.11 is the
  optimistic reading against those two references.

## Files

- `figures/fig_benchmark.svg`, `figures/fig_shrinkage.svg` — produced by
  `python/build_result_figures.py` (run from a checkout with both branches' outputs on disk;
  paths at the top of the script)
- `figures/fig_pipeline.svg` / `.png` — the hand-authored two-lane pipeline diagram. **No longer in
  the deck** (its content moved into the columns of slide 1); kept as a standalone diagram.
- `structures`: the 3D panels use the protein atoms of
  `outputs/report/structures/1M6O.pdb` from `foundation-models-testing` — chain A trimmed to
  residues 1-182 for the left panel (the alpha1/alpha2 construct that was folded), all three chains
  for the right. Experimental coordinates; nothing on that slide is predicted.
- `SPEAKER_NOTES.md` — per-slide talking points and the statistics to have ready
