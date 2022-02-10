from pprint import pprint
import unittest

from a2t.pipeline.candidates import CandidateGenerator, UnaryToBinaryCandidateGenerator
from a2t.pipeline.base import Pipeline
from a2t.pipeline.filters import NegativeInstanceFilter
from a2t.tasks import UnaryFeatures, BinaryFeatures, NamedEntityClassificationTask, BinaryTask
from a2t.tasks.span_classification import NamedEntityClassificationFeatures
from a2t.tasks.tuple_classification import RelationClassificationFeatures, RelationClassificationTask


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

    def test_group_features(self):

        unary_features = [
            UnaryFeatures(context=f"An example context. {j}", X=str(i), label="NUMBER") for i in range(3) for j in range(3)
        ]

        grouped_features = CandidateGenerator.group_features(unary_features, by="context")

        for j, group in zip(range(3), grouped_features):
            self.assertListEqual(unary_features[j::3], group)


class TestPipelines(unittest.TestCase):
    def test_pipeline(self):

        ner_labels = ["O", "Person", "Location", "Organization"]

        re_labels = ["no_relation", "per:city_of_death", "org:founded_by"]
        re_templates = {
            "per:city_of_death": ["{X} died in {Y}."],
            "org:founded_by": ["{X} was founded by {Y}.", "{Y} founded {X}."],
        }
        re_valid_conditions = {
            "per:city_of_death": ["Person:Location"],
            "org:founded_by": ["Organization:Person"],
        }

        pipe = Pipeline(
            (NamedEntityClassificationTask("NER", labels=ner_labels), "initial_features", "ner_features"),
            (
                UnaryToBinaryCandidateGenerator(NamedEntityClassificationFeatures, RelationClassificationFeatures),
                "ner_features",
                "re_features",
            ),
            (NegativeInstanceFilter("O"), "ner_features", "ner_features"),
            (
                RelationClassificationTask(
                    "RE",
                    labels=re_labels,
                    templates=re_templates,
                    valid_conditions=re_valid_conditions,
                ),
                "re_features",
                "re_features",
            ),
            (NegativeInstanceFilter("no_relation"), "re_features", "re_features"),
            threshold=0.5,
            NER_model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            RE_threshold=0.9,
            use_tqdm=False
        )

        features = [
            NamedEntityClassificationFeatures(
                context="Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
                label=None,
                inst_type="NNP",
                X="Billy Mays",
            ),
            NamedEntityClassificationFeatures(
                context="Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
                label=None,
                inst_type="NNP",
                X="Tampa",
            ),
            NamedEntityClassificationFeatures(
                context="Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
                label=None,
                inst_type="NNP",
                X="Sunday",
            ),
        ]

        result = pipe(features)
        
        self.assertListEqual(["Person", "Location"], [feature.label for feature in result["ner_features"]])
        self.assertListEqual(["per:city_of_death"], [feature.label for feature in result["re_features"]])
