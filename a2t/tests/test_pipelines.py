import unittest

from a2t.pipeline.candidates import UnaryToBinaryCandidateGenerator
from a2t.tasks import UnaryFeatures, BinaryFeatures


class TestCandidateGeneration(unittest.TestCase):
    def test_unary_to_binary_candidate_generation(self):

        unary_features = [UnaryFeatures(context="An example context.", X=str(i), label="NUMBER") for i in range(3)]
        binary_features = [
            BinaryFeatures(context="An example context.", X="0", Y="1", inst_type="NUMBER:NUMBER", label=None),
            BinaryFeatures(context="An example context.", X="0", Y="2", inst_type="NUMBER:NUMBER", label=None),
            BinaryFeatures(context="An example context.", X="1", Y="0", inst_type="NUMBER:NUMBER", label=None),
            BinaryFeatures(context="An example context.", X="1", Y="2", inst_type="NUMBER:NUMBER", label=None),
            BinaryFeatures(context="An example context.", X="2", Y="0", inst_type="NUMBER:NUMBER", label=None),
            BinaryFeatures(context="An example context.", X="2", Y="1", inst_type="NUMBER:NUMBER", label=None),
        ]

        generator = UnaryToBinaryCandidateGenerator()

        self.assertListEqual(binary_features, list(generator(unary_features)))
