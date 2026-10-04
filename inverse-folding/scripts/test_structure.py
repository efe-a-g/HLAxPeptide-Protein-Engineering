"""Independent, small structural checks; no complete experiment is repeated.

Requires the saved LigandMPNN checkpoint and downloaded 1HHK template. Uses one
191-residue context and a handful of synthetic peptides, normally a few seconds.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from structural_pipeline import (AA, SEEDS, Scorer, build_template, cache_path,
                                 encode_seq, features_from_logps)
import structural_pipeline as structural
from splits import load_data


class StructuralDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.scorer = Scorer(threads=1)
        cls.template = build_template("1HHK", "truncated", cls.data)
        cls.hla = cls.data.loc[cls.data.allele == "HLA-A*02:01", "hla_seq"].iloc[0]
        cls.context = cls.scorer.prepare(cls.template, cls.hla, SEEDS[0])

    def test_cached_decoder_matches_official_and_has_correct_context(self):
        result = self.scorer.validate(self.template, self.hla)
        self.assertTrue(result["passed"])
        self.assertLess(result["conditional_official_max_abs_error"], 3e-5)
        self.assertLess(result["unconditional_nine_official_calls_max_abs_error"], 3e-5)
        self.assertLess(result["unconditional_peptide_mutation_max_abs_error"], 2e-6)
        self.assertGreater(result["unconditional_hla_shuffle_max_abs_change"], 1e-4)

    def test_autoregressive_predictions_cannot_read_self_or_future_identity(self):
        seed = self.context["seed"]
        order = np.random.default_rng(seed + 9001).permutation(9)
        first, last = int(order[0]), int(order[-1])
        original = "ACDEFGHIK"

        def mutate(position):
            seq = list(original)
            seq[position] = "W" if seq[position] != "W" else "A"
            return "".join(seq)

        values = self.scorer.score(self.context, [original, mutate(last), mutate(first)], "conditional")
        # Last-decoded identity cannot affect any prediction, including its own.
        np.testing.assert_allclose(values[0], values[1], rtol=0, atol=2e-6)
        # First-decoded identity cannot affect its own prediction, but should affect
        # at least one subsequent prediction through allowed peptide context.
        np.testing.assert_allclose(values[0, first], values[2, first], rtol=0, atol=2e-6)
        later = order[1:]
        self.assertGreater(float(np.abs(values[0, later] - values[2, later]).max()), 1e-4)

    def test_preparing_other_hla_cannot_mutate_cached_context(self):
        before = self.scorer.score(self.context, ["ACDEFGHIK"], "unconditional")
        self.scorer.prepare(self.template, self.hla, SEEDS[0], scrambled=True)
        after = self.scorer.score(self.context, ["ACDEFGHIK"], "unconditional")
        np.testing.assert_array_equal(before, after)

    def test_cache_keys_separate_every_material_input(self):
        meta = self.template[-1]
        default = dict(scorer=self.scorer, meta=meta, allele="HLA-A*02:01", hla=self.hla,
                       mode="conditional", seed=SEEDS[0], scrambled=False, peptides=["ACDEFGHIK"])
        path, key = cache_path(**default)
        same, _ = cache_path(**copy.copy(default))
        self.assertEqual(path, same)
        alternatives = [dict(seed=29), dict(scrambled=True), dict(mode="unconditional"),
                        dict(allele="HLA-A*03:01"), dict(hla=self.hla[::-1]),
                        dict(peptides=["ACDEFGHIW"]), dict(peptides=["ACDEFGHIK", "ACDEFGHIW"]),
                        dict(meta={**meta, "template_sha256": "different_geometry"})]
        for change in alternatives:
            changed, _ = cache_path(**(default | change))
            self.assertNotEqual(path, changed, change)
        self.assertIn("scoring_algorithm_sha256", key)
        self.assertIn("checkpoint", key)
        self.assertIn("source_commit", key)

    def test_conditional_cache_grows_and_preserves_peptide_alignment(self):
        first = pd.DataFrame({"row_id": [17, 42], "allele": ["HLA-A*02:01"]*2,
                              "hla_seq": [self.hla]*2, "peptide": ["ACDEFGHIK", "LMNPQRSTV"]})
        extended = pd.concat([first.iloc[[1]], first.iloc[[0]].assign(row_id=57, peptide="WWWWWWWWW"), first.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory(prefix=".test_structure_", dir=structural.ROOT) as directory:
            with patch.object(structural, "OUT", Path(directory)), patch.object(structural, "SEEDS", [SEEDS[0]]):
                with patch.object(self.scorer, "score", wraps=self.scorer.score) as score:
                    base, _, _ = structural.compute(self.scorer, first, [self.template], "conditional")
                    grown, _, _ = structural.compute(self.scorer, extended, [self.template], "conditional")
                    cached, _, _ = structural.compute(self.scorer, extended, [self.template], "conditional")
                    self.assertEqual(score.call_count, 2, "Only initial and newly missing peptides should be scored")
                    self.assertEqual(score.call_args_list[0].args[1], first.peptide.tolist())
                    self.assertEqual(score.call_args_list[1].args[1], ["WWWWWWWWW"])
                np.testing.assert_array_equal(base[:, 0], grown[:, 2])
                np.testing.assert_array_equal(base[:, 1], grown[:, 0])
                np.testing.assert_array_equal(grown, cached)


class StructuralFeaturesTests(unittest.TestCase):
    def test_average_observed_log_likelihood_and_normalized_anchor_features(self):
        data = pd.DataFrame({"row_id": [17, 42], "peptide": ["AAAAAAAAA", "CCCCCCCCC"]})
        probabilities = np.full((2, 2, 9, 20), 1 / 20, dtype=np.float64)
        for ensemble, observed_probability in enumerate([.8, .2]):
            for row, residue in enumerate(["A", "C"]):
                probabilities[ensemble, row] = (1 - observed_probability) / 19
                probabilities[ensemble, row, :, AA.index(residue)] = observed_probability
        result = features_from_logps(data, np.log(probabilities))
        # Arithmetic averaging of log scores differs from log of averaged p.
        expected = (np.log(.8) + np.log(.2)) / 2
        np.testing.assert_allclose(result["mean_logp"], expected)
        self.assertGreater(abs(expected - np.log(.5)), .1)
        np.testing.assert_allclose(result["logp_per_position"], expected)
        anchors = result["anchor_softmax"].reshape(2, 2, 20)
        np.testing.assert_allclose(anchors.sum(-1), 1)
        self.assertAlmostEqual(float(anchors[0, 0, AA.index("A")]), .5)
        np.testing.assert_allclose(result["spread"], np.std([np.log(.8), np.log(.2)]))
        np.testing.assert_array_equal(result["row_id"], [17, 42])


if __name__ == "__main__":
    unittest.main()
