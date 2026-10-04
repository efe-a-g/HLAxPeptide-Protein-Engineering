# slides — the two foundation-model routes, as a talk

`foundation_model_routes_deck.html` is a **self-contained** reveal.js 6.0.2 deck: the reveal
CSS/JS, the white theme (with its fonts) and all three figures are inlined as data URIs, so it
opens from `file://` with no server and no network. Arrow keys advance; `down` on slide 2 reveals
the two backup tables; `F` fullscreen, `O` overview.

Four slides: (1) the pipeline graphic, (2) two spinning 3D cartoons of what each route is actually
shown, (3) both routes against the common benchmark, (4) why each closed plus the few-allele
inflation result.

The 3D slide embeds 3Dmol.js 2.5.5 (inlined) and the protein atoms of
`outputs/report/structures/1M6O.pdb` from `foundation-models-testing` -- chain A trimmed to
residues 1-182 for the Route A panel (the alpha1/alpha2 construct that was folded), all three
chains for Route B. Peptide in amber, one residue paler to echo the low-confidence residue in the
pipeline graphic. Nothing on that slide is a prediction.

## Where the numbers come from

- Route A arms: `outputs/supertype/results.jsonl` on `foundation-models-testing`
  (canonical supertype leave-allele-out, 5 seeds). BLOSUM baseline 0.4626 +/- 0.0379,
  peptide-only null 0.3045, best embedding arm 0.4206.
- Route B predictors: `confidence/outputs/predictAll_predictor_comparison.csv`
  (leave-one-allele-out, 67 alleles, 1,430 co-folded trimers). Best arm 0.1119 +/- 0.0259.
- Stated on the slide, not buried: Route B's leave-one-allele-out is the *easier* split, so 0.11
  is the optimistic reading against those two references.

## Files

- `figures/fig_pipeline.svg` — hand-authored SVG (edit it directly; `fig_pipeline.png` is a 2560 px raster of it)
- `figures/fig_benchmark.svg`, `figures/fig_shrinkage.svg` — produced by `python/build_result_figures.py`
  (run from a checkout that has both branches' output files on disk; paths at the top of the script)
- `SPEAKER_NOTES.md` — per-slide talking points and the statistics to have ready
