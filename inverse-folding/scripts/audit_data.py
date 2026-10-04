#!/usr/bin/env python3
"""Reproducible, dependency-free Phase 0 audit. All motif fits are descriptive.

These full-data motifs must never be reused as supervised evaluation features.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import pathlib
import statistics

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_PSEUDO_ONE_BASED = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
                            84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]


def quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)


def distribution(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "quantiles": {str(p): quantile(values, p) for p in [0, .01, .05, .10, .25, .5, .75, .9, .95, .99, 1]}}


def pearson(x, y):
    if len(x) < 2:
        return None
    xm, ym = statistics.mean(x), statistics.mean(y)
    xd, yd = [v - xm for v in x], [v - ym for v in y]
    denominator = math.sqrt(sum(v * v for v in xd) * sum(v * v for v in yd))
    return sum(a * b for a, b in zip(xd, yd)) / denominator if denominator else None


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        for i in order[left:right]:
            result[i] = (left + right - 1) / 2 + 1
        left = right
    return result


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def js_divergence(p, q):
    midpoint = [(a + b) / 2 for a, b in zip(p, q)]
    return sum(a * math.log2(a / m) / 2 if a else 0 for a, m in zip(p, midpoint)) + sum(b * math.log2(b / m) / 2 if b else 0 for b, m in zip(q, midpoint))


def mapping_audit(allele_sequences):
    alleles = sorted(allele_sequences)
    columns = [tuple(allele_sequences[a][0][i] for a in alleles) for i in range(182)]
    candidates = [[i for i, column in enumerate(columns)
                   if column == tuple(allele_sequences[a][1][j] for a in alleles)]
                  for j in range(34)]
    solutions = []

    def extend(prefix):
        if len(prefix) == 34:
            solutions.append(prefix)
            return
        for i in candidates[len(prefix)]:
            if not prefix or i > prefix[-1]:
                extend(prefix + [i])

    extend([])
    assert all("".join(sequence[i] for i in solution) == pseudo
               for solution in solutions for sequence, pseudo in allele_sequences.values())
    union = sorted(set(itertools.chain.from_iterable(solutions)))
    intersection = sorted(set(solutions[0]).intersection(*map(set, solutions[1:]))) if solutions else []
    canonical = [i - 1 for i in CANONICAL_PSEUDO_ONE_BASED]
    assert canonical in solutions, "Published canonical mapping does not match this data"
    return {
        "indexing": "zero-based hla_seq positions; assumed subsequence order strictly increasing",
        "n_alleles_checked": len(alleles), "joint_column_candidates_zero_based": candidates,
        "n_ordered_solutions": len(solutions), "unique": len(solutions) == 1,
        "all_ordered_solutions_zero_based": solutions,
        "all_ordered_solutions_one_based": [[i + 1 for i in solution] for solution in solutions],
        "certain_indices_zero_based": intersection, "possible_indices_zero_based": union,
        "ambiguous_pseudo_positions_one_based": [j + 1 for j in range(34) if len({s[j] for s in solutions}) > 1],
        "representative_indices_zero_based": solutions[0] if solutions else None,
        "representative_warning": "Lexicographically first valid set, not a uniquely recovered or externally validated mapping.",
        "canonical_indices_zero_based": canonical,
        "canonical_indices_one_based": CANONICAL_PSEUDO_ONE_BASED,
        "canonical_selection_note": "Standard NetMHCpan positions, selected using prior biological annotation; matches all rows but is not uniquely identifiable from this table alone.",
        "pseudosequence_source_urls": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC1949492/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4498290/"],
        "clamp_positions_one_based": {str(i): {"present_in_every_solution": i - 1 in intersection,
                                               "residues_in_hla_seq": sorted({s[0][i - 1] for s in allele_sequences.values()})}
                                       for i in [7, 59, 159, 171, 143, 146, 147]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("resources/rasmussen_et_al_dataset - rasmussen_et_al_dataset.csv"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/audit"))
    parser.add_argument("--epsilon", type=float, default=.05)
    parser.add_argument("--motif-pseudocount", type=float, default=.5)
    args = parser.parse_args()
    assert args.epsilon > 0 and args.motif_pseudocount > 0
    args.out.mkdir(parents=True, exist_ok=True)
    with args.data.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"allele", "peptide", "thalf_hours", "hla_seq", "hla_pseudoseq"}
    assert rows and required <= set(rows[0]), "Missing required columns"
    by_allele, by_pair = collections.defaultdict(list), collections.defaultdict(list)
    by_peptide = collections.defaultdict(set)
    allele_sequences = {}
    for row_id, row in enumerate(rows):
        row["target"] = float(row["thalf_hours"])
        row["row_id"] = row_id
        assert math.isfinite(row["target"]) and row["target"] >= 0
        assert len(row["peptide"]) == 9 and set(row["peptide"]) <= set(AMINO_ACIDS)
        assert len(row["hla_seq"]) == 182 and len(row["hla_pseudoseq"]) == 34
        assert set(row["hla_seq"] + row["hla_pseudoseq"]) <= set(AMINO_ACIDS)
        seq_pair = row["hla_seq"], row["hla_pseudoseq"]
        assert row["allele"] not in allele_sequences or allele_sequences[row["allele"]] == seq_pair
        allele_sequences[row["allele"]] = seq_pair
        by_allele[row["allele"]].append(row)
        by_pair[row["allele"], row["peptide"]].append(row)
        by_peptide[row["peptide"]].add(row["allele"])
    targets = [r["target"] for r in rows]
    positive = [y for y in targets if y > 0]
    zeros = sum(y == 0 for y in targets)
    zero_rate = zeros / len(rows)
    allele_summary, composition_rows, composition_tests, motifs = [], [], [], {}
    for allele, allele_rows in sorted(by_allele.items()):
        ys = [r["target"] for r in allele_rows]
        n0 = sum(y == 0 for y in ys)
        stable_rows = [r for r in allele_rows if r["target"] > 2]
        allele_summary.append({"allele": allele, "n": len(ys), "n_zero": n0, "zero_fraction": n0 / len(ys),
                               "n_positive": len(ys) - n0, "n_stable_gt_2h": len(stable_rows),
                               "stable_fraction": len(stable_rows) / len(ys),
                               "mean_hours": statistics.mean(ys), "median_hours": statistics.median(ys),
                               "mean_log_target": statistics.mean(math.log(y + args.epsilon) for y in ys),
                               "small_sample_n_lt_50": len(ys) < 50,
                               "few_stable_n_lt_20": len(stable_rows) < 20})
        counts = [[sum(r["peptide"][i] == aa for r in stable_rows) for aa in AMINO_ACIDS] for i in range(9)]
        probabilities = [[(c + args.motif_pseudocount) / (len(stable_rows) + 20 * args.motif_pseudocount) for c in position] for position in counts]
        motifs[allele] = {"n_stable": len(stable_rows), "counts": counts, "probabilities": probabilities}
    # Both pooled and stratified anchor composition avoid confounding by allele mix.
    for allele, allele_rows in [("__pooled__", rows)] + list(sorted(by_allele.items())):
        for position in [2, 9]:
            vectors = {}
            for group, subset in [("zero", [r for r in allele_rows if r["target"] == 0]),
                                  ("positive", [r for r in allele_rows if r["target"] > 0]),
                                  ("stable_gt_2h", [r for r in allele_rows if r["target"] > 2])]:
                counts = collections.Counter(r["peptide"][position - 1] for r in subset)
                vectors[group] = [counts[aa] / len(subset) if subset else 0 for aa in AMINO_ACIDS]
                for aa in AMINO_ACIDS:
                    composition_rows.append({"allele": allele, "position": position, "group": group,
                                             "amino_acid": aa, "count": counts[aa], "n_group": len(subset),
                                             "frequency": counts[aa] / len(subset) if subset else None})
            n0 = sum(r["target"] == 0 for r in allele_rows)
            npos = len(allele_rows) - n0
            composition_tests.append({"allele": allele, "position": position, "n_zero": n0, "n_positive": npos,
                                      "zero_positive_js_divergence_bits": js_divergence(vectors["zero"], vectors["positive"]) if n0 and npos else None})
    repeated = {pair: rs for pair, rs in by_pair.items() if len(rs) > 1}
    replicate_pairs = [(pair, a, b) for pair, rs in repeated.items() for a, b in itertools.combinations(rs, 2)]
    rx, ry = [math.log(a["target"] + args.epsilon) for _, a, _ in replicate_pairs], [math.log(b["target"] + args.epsilon) for _, _, b in replicate_pairs]
    rep_r = pearson(rx, ry)
    replicate_report = {"n_unique_allele_peptide_pairs": len(by_pair), "n_repeated_pairs": len(repeated),
                        "n_rows_in_repeated_pairs": sum(map(len, repeated.values())),
                        "n_pairwise_replicate_comparisons": len(replicate_pairs),
                        "pearson_log_target": rep_r, "spearman": pearson(ranks(rx), ranks(ry)),
                        "estimated_single_measurement_reliability": rep_r,
                        "ideal_latent_predictor_pearson_ceiling_sqrt_reliability": math.sqrt(rep_r) if rep_r is not None and rep_r >= 0 else None,
                        "noise_ceiling_status": "not_estimable_no_replicates" if not repeated else "replicate_estimate_requires_independent_exchangeable_errors",
                        "plot_annotation": "Replicate noise ceiling: unavailable (no repeated allele-peptide pairs)" if not repeated else "Replicate agreement is an empirical reference, not an unconditional prediction bound",
                        "limitations": ["Repeated peptides across different alleles are different biological complexes and cannot estimate replicate noise.",
                                        "The input may have already averaged or deduplicated experimental replicates; their raw measurements are unavailable.",
                                        "Replicate Pearson is reliability under independent exchangeable additive errors; sqrt(reliability), not reliability, bounds an ideal latent predictor's correlation with one noisy outcome under those assumptions.",
                                        "Spearman replicate agreement is not a universal Spearman ceiling."]}
    n = len(rows)
    chi2 = sum((a["n_zero"] - a["n"] * zero_rate) ** 2 / (a["n"] * zero_rate) +
               (a["n_positive"] - a["n"] * (1 - zero_rate)) ** 2 / (a["n"] * (1 - zero_rate)) for a in allele_summary)
    mapping = mapping_audit(allele_sequences)
    motif_checks = {}
    for allele, position, preferred in [("HLA-A*02:01", 2, "LM"), ("HLA-A*02:01", 9, "VL"), ("HLA-B*27:05", 2, "R"), ("HLA-A*03:01", 9, "KR")]:
        m = motifs[allele]
        numerator = sum(m["counts"][position - 1][AMINO_ACIDS.index(aa)] for aa in preferred)
        key = f"{allele}_P{position}_{preferred}"
        motif_checks[key] = {"n_stable": m["n_stable"], "preferred_count": numerator,
                             "preferred_fraction_unsmoothed": numerator / m["n_stable"] if m["n_stable"] else None}
        for group, subset in [("zero", [r for r in by_allele[allele] if r["target"] == 0]),
                              ("positive", [r for r in by_allele[allele] if r["target"] > 0])]:
            preferred_count = sum(r["peptide"][position - 1] in preferred for r in subset)
            motif_checks[key][group] = {"n": len(subset), "preferred_count": preferred_count,
                                       "preferred_fraction": preferred_count / len(subset) if subset else None}
    pooled_mean_log = statistics.mean(math.log(y + args.epsilon) for y in targets)
    total_ss = sum((math.log(y + args.epsilon) - pooled_mean_log) ** 2 for y in targets)
    between_ss = sum(a["n"] * (a["mean_log_target"] - pooled_mean_log) ** 2 for a in allele_summary)
    zero_by_allele = sorted(allele_summary, key=lambda a: a["zero_fraction"])
    report = {"source": str(args.data), "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "config": {"epsilon_hours": args.epsilon, "epsilon_note": "Computational offset; minimum observed positive is not an established assay censoring threshold.",
                         "motif_stability_threshold_hours": 2, "motif_pseudocount_per_amino_acid": args.motif_pseudocount,
                         "motif_amino_acid_order": AMINO_ACIDS},
              "n_rows": n, "n_alleles": len(by_allele), "n_unique_peptides": len(by_peptide),
              "n_peptides_seen_with_multiple_alleles": sum(len(alleles) > 1 for alleles in by_peptide.values()),
              "n_rows_whose_peptide_seen_with_multiple_alleles": sum(len(by_peptide[r["peptide"]]) > 1 for r in rows),
              "target_hours_all": distribution(targets), "target_hours_positive": distribution(positive),
              "log_target_all": distribution([math.log(y + args.epsilon) for y in targets]),
              "exact_zeros": zeros, "zero_fraction": zero_rate, "minimum_positive_hours": min(positive),
              "near_zero_counts": {str(t): {"zero_inclusive_count_le_threshold": sum(y <= t for y in targets),
                                              "positive_count_le_threshold": sum(0 < y <= t for y in targets)} for t in [.01, .05, .1, .2, .5]},
              "zeros_allele_association": {"pearson_chi_square": chi2, "degrees_freedom": len(by_allele) - 1,
                                           "cramers_v": math.sqrt(chi2 / n), "p_value": None,
                                           "p_value_note": "Effect size provided; no asymptotic p-value computed by dependency-free audit.",
                                           "min_zero_fraction": zero_by_allele[0], "max_zero_fraction": zero_by_allele[-1]},
              "allele_imbalance": {"minimum_n": min(map(len, by_allele.values())), "maximum_n": max(map(len, by_allele.values())),
                                    "largest_to_smallest_ratio": max(map(len, by_allele.values())) / min(map(len, by_allele.values())),
                                    "n_alleles_below_50": sum(len(v) < 50 for v in by_allele.values()),
                                    "n_alleles_below_100": sum(len(v) < 100 for v in by_allele.values()),
                                    "n_alleles_with_fewer_than_20_stable": sum(a["n_stable_gt_2h"] < 20 for a in allele_summary)},
              "between_allele_fraction_of_log_target_variation_descriptive": between_ss / total_ss,
              "replicates": replicate_report, "pseudosequence_mapping": mapping,
              "motif_biology_checks": motif_checks,
              "source_context": {"paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/",
                                 "measurement_note": "Paper describes 28,166 measured points, each a geometric mean of two experiments, before separate augmentation with 75,000 assigned-zero weak binders. This table has 28,166 rows and likely represents the measured table; it does not expose raw experimental replicates.",
                                 "provenance_note": "Matching row count is evidence of dataset identity, not per-row proof of provenance."},
              "interpretation": ["A 20% point mass at exactly zero is present. It is not sufficient evidence of synthetic weak-binder augmentation: the paper reports 28,166 measured points before adding 75,000 weak-binder zero labels. Treat zeros as a censored sensitivity analysis; the assay resolution is not identified by this table.",
                                 "Compare zero-positive anchor composition within allele; pooled composition can reflect different allele mixtures.",
                                 "All data-derived ordered pseudosequence mappings include positions 7/59/159/171 and 143/147; the assertion that these clamp positions are absent is contradicted by this table. Some alleged invariant positions vary in this dataset; see residue sets in mapping audit.",
                                 "Invariant sequence identities cannot distinguish alleles. Structural geometry might still add information, but conserved-clamp omission is not established here.",
                                 "Empirical motifs use all rows for descriptive structural comparisons only. Fit supervised PWM features on training rows separately."]}
    write_json(args.out / "audit.json", report)
    write_json(args.out / "pseudosequence_indices.json", mapping)
    write_json(args.out / "pseudosequence_indices_representative.json", mapping["representative_indices_zero_based"])
    write_json(args.out / "pseudosequence_indices_candidates.json", mapping["all_ordered_solutions_zero_based"])
    write_json(args.out / "pseudosequence_indices_canonical.json", mapping["canonical_indices_zero_based"])
    write_json(args.out / "replicate_noise_ceiling.json", replicate_report)
    write_json(args.out / "empirical_motifs.json", {"amino_acids": AMINO_ACIDS, "positions": list(range(1, 10)),
                                                 "threshold_hours_exclusive": 2, "pseudocount_per_amino_acid": args.motif_pseudocount,
                                                 "usage": "Descriptive full-data motifs; not training or evaluation features", "alleles": motifs})
    write_csv(args.out / "allele_summary.csv", sorted(allele_summary, key=lambda a: -a["n"]))
    write_csv(args.out / "zero_anchor_composition.csv", composition_rows)
    write_csv(args.out / "zero_anchor_divergence.csv", composition_tests)
    write_csv(args.out / "replicate_pairs.csv", [{"allele": p[0], "peptide": p[1], "row_id_a": a["row_id"], "row_id_b": b["row_id"], "target_a": a["target"], "target_b": b["target"]} for p, a, b in replicate_pairs],
              ["allele", "peptide", "row_id_a", "row_id_b", "target_a", "target_b"])
    write_csv(args.out / "empirical_motifs.csv", ({"allele": allele, "n_stable": motif["n_stable"], "position": pos + 1,
                                                "amino_acid": aa, "count": motif["counts"][pos][j], "probability": motif["probabilities"][pos][j]}
                                               for allele, motif in motifs.items() for pos in range(9) for j, aa in enumerate(AMINO_ACIDS)))
    write_csv(args.out / "target_frequency.csv", ({"thalf_hours": y, "count": count} for y, count in sorted(collections.Counter(targets).items())))
    summary = {key: report[key] for key in ["n_rows", "n_alleles", "n_unique_peptides", "exact_zeros", "zero_fraction", "minimum_positive_hours", "near_zero_counts", "allele_imbalance", "motif_biology_checks"]}
    summary["n_repeated_pairs"] = len(repeated)
    summary["n_pseudosequence_index_solutions"] = mapping["n_ordered_solutions"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
