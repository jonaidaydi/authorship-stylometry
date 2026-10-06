"""Small semantic checks for the experiment's data and distance conventions."""

import unittest

import numpy as np
import spacy

from stylometry.features import clean_text, extract_document_features
from stylometry.federalist import parse_federalist, roman_number
from stylometry.models import fit_scaler, nearest, transform_style


class MethodTests(unittest.TestCase):
    def test_style_counts_and_sentence_units(self):
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        features = extract_document_features(nlp("The cat, and the dog. A bird!"))
        self.assertEqual(len(features), 26)
        self.assertEqual(features["mean_sentence_length"], 3.5)
        self.assertEqual(features["sentence_length_std"], 1.5)
        self.assertEqual(features["function_the_per_1000"], 285.714)
        self.assertEqual(features["punctuation_comma_per_1000"], 142.857)
        self.assertTrue(all(value == 0 for value in extract_document_features(nlp("")).values()))

    def test_reference_scaler_does_not_fit_to_queries(self):
        mean, std, variable = fit_scaler(np.array([[0., 5.], [2., 5.]]))
        np.testing.assert_array_equal(variable, [True, False])
        np.testing.assert_array_equal(mean, [1., 5.])
        np.testing.assert_array_equal(transform_style(np.array([[101., -9.]]), mean, std, variable), [[100.]])

    def test_nearest_and_deterministic_tie(self):
        labels, distances, indices = nearest(np.array([[0., 0.], [2., 0.]]), np.array([[1., 0.], [2., 1.]]), ["A", "B"])
        np.testing.assert_array_equal(labels, ["A", "B"])
        np.testing.assert_allclose(distances, [1., 1.])
        np.testing.assert_array_equal(indices, [0, 1])

    def test_minimal_cleaning(self):
        self.assertEqual(clean_text(" A &amp; B!\r\n "), "A & B!")

    def test_federalist_metadata_and_duplicate(self):
        blocks = []
        for number in range(1, 86):
            block = f"THE FEDERALIST.\nNo. {number}.\nTitle\nHAMILTON\nTo the People of the State of New York:\n" + "Essay body with enough words. " * 10 + "\nPUBLIUS.\nEditorial note.\n"
            blocks.append(block)
            if number == 70:
                blocks.append(block.replace("Essay body", "Alternative body"))
        records = parse_federalist("\n".join(blocks))
        self.assertEqual(len(records), 85)
        self.assertEqual(records[69]["number"], 70)
        self.assertTrue(records[69]["body"].startswith("Essay body"))
        self.assertNotIn("HAMILTON", records[0]["body"])
        self.assertNotIn("Editorial", records[0]["body"])
        self.assertEqual(records[48]["author"], "Disputed")
        self.assertEqual(records[17]["author"], "Joint")
        self.assertEqual(roman_number("LIX"), 59)


if __name__ == "__main__":
    unittest.main()
