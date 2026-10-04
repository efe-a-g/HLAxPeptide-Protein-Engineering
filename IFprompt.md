## Inverse folding (primary differentiated experiment)

Rationale: inverse-folding models give p(sequence | backbone). Conditioning on
the groove geometry and scoring the observed peptide yields a zero-shot
stability proxy that uses the conserved clamp and pocket geometry the
pseudosequence discards. No training, no labels, so immune to every split
concern, and naturally pan-specific.

Use **LigandMPNN** (better atomic context handling, on Serova's model list);
fall back to ProteinMPNN if setup is slow.

### Template construction
- Pull 5-10 high-resolution class I structures with **9-mer** peptides from the
  PDB, spanning different alleles. (PDB 4MJI, cited in the brief, is an 8-mer on
  B*51:01 — do not use it as the sole template.)
- Strip to backbone (N, CA, C, O). HLA and peptide as separate chains in one
  file — the peptide must be a spatially adjacent chain, not appended to the HLA
  sequence.
- Decide empirically whether to truncate the HLA to alpha1/alpha2 (matches our
  data exactly, but leaves an unnatural exposed interface where alpha3 was) or
  keep the full assembly with alpha3 and beta2-microglobulin. **Test both on a
  500-row subset before committing** — this can quietly degrade everything
  downstream.

### Two scoring modes, both required — they are different experiments

**(b) UNCONDITIONAL — run this first.** Peptide sequence is NOT an input. One
forward pass per allele (~75 total) over backbone + fixed `hla_seq`, giving a
9x20 matrix per allele: the model's geometry-derived prediction of what each
groove wants at each position. Score any peptide by lookup and sum. Assumes
positional independence. ~400x cheaper than (a), essentially free.

Immediately plot these predicted motifs against the empirical motifs from
Phase 0.5. **Does structure alone recover B*27:05's P2 arginine preference?**
This is the single most convincing figure if it works, and tells you within an
hour whether the structural channel carries real signal. If these motifs are
garbage, (a) will not rescue the approach.

**(a) CONDITIONAL (autoregressive).** Peptide sequence IS an input. Per row:
`sum_i log p(x_i | x_<i, backbone, hla_seq)`. Captures inter-position
dependence — how the choice at P2 shifts what is tolerated at P9. 28k forward
passes, minutes of compute.

Setup: `--chain_id_jsonl` marks the HLA chain fixed and the peptide designable;
`--score_only` evaluates the supplied sequence rather than sampling a new one.
Decoding order is randomised, so scores wobble between runs — average over
several random orders and over the template ensemble, and report the spread, not
just a point estimate. Verify current flag names against the repo rather than
trusting these.

**Report the gap between (a) and (b).** If (a) barely beats (b), class I binding
is close to positionally independent — the implicit assumption behind every PWM
in the field, here tested directly against a model capable of violating it.
That is a result worth stating either way.

### Features extracted
From the same forward passes:
- `mpnn_mean_logp` — scalar.
- `mpnn_logp_per_position` — 9 values. Expected to beat the scalar, since a sum
  dilutes the P2/P9 signal across seven positions that barely matter.
- `mpnn_anchor_softmax` — full 20-way distributions at P2 and P9 (40 values).
  Likely the most informative: the model's statement of what the pocket wants,
  which is what the pseudosequence is a proxy for, arrived at independently.

Include all three and ablate; the marginal cost is zero.

### Mandatory controls
- **Scrambled-HLA control.** Re-score with `hla_seq` shuffled. If scores barely
  move, the model is reading peptide sequence alone and the structural framing
  is decoration. **Run this before building anything on top.**
- **Per-position log-prob profile.** Plot mean log-prob across P1-P9. Should
  show the model most constrained at P2 and P9. A flat profile means anchor
  chemistry isn't being captured.
- Where real PDB structures exist for our alleles, compare shared-template
  scoring against per-allele structures on those alleles. Justifies the shortcut
  with evidence rather than assertion.

### Reporting
Three rows in the results table:
1. Structural score alone, zero-shot, no training.
2. Phase 2 baseline alone.
3. Baseline + structural features.

Row 3 is only interesting relative to row 2 — it tests whether the structural
information is orthogonal to the hand-crafted encoding. If it adds nothing, the
information was already there, which is a clean result with a mechanism.

## Engineering

- Seed everything; log all runs with config.
- Cache MPNN scores to disk keyed by (template, allele, peptide, seed) — do not
  recompute.
- Build the full pipeline on a 1,000-row stratified subset first (uniform
  sampling gives mostly A*02:01), verify end to end, then scale.
- Standardise structural features before concatenation — they will otherwise be
  swamped by an ~860-dim one-hot baseline vector. Consider a separate branch
  merged late in the head.
