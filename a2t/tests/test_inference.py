import unittest

from a2t.base import EntailmentClassifier
from a2t.tasks.text_classification import TopicClassificationFeatures, TopicClassificationTask

import os

from a2t.tasks.tuple_classification import RelationClassificationFeatures, RelationClassificationTask

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TestEntailmentModel(unittest.TestCase):
    def test_load_model(self):

        # Create a task
        task = TopicClassificationTask(
            "DummyTopic task", labels=["politics", "culture", "economy", "biology", "legal", "medicine", "business"]
        )

        features = [
            TopicClassificationFeatures(
                context="hospital: a health facility where patients receive treatment.", label="medicine"
            )
        ]

        nlp = EntailmentClassifier(use_tqdm=False)

        preds = nlp(task=task, features=features, negative_threshold=0.0, return_confidences=True, return_labels=True, topk=3)

        for (pred_label, pred_prob), (gold_label, gold_prob) in zip(
            preds[0], [("medicine", 0.8547821), ("biology", 0.036895804), ("business", 0.032091234)]
        ):
            self.assertEqual(pred_label, gold_label)
            self.assertAlmostEqual(pred_prob, gold_prob, places=2)


    def test_empty_templates(self):
        labels = ["no_relation", "per:city_of_death", "org:founded_by"]

        templates = {
            "per:city_of_death": [], # Emtpty list
            "org:founded_by": ["{X} was founded by {Y}.", "{Y} founded {X}."],
        }

        valid_conditions = {"per:city_of_death": ["PERSON:CITY", "PERSON:LOCATION"], "org:founded_by": ["ORGANIZATION:PERSON"]}

        task = RelationClassificationTask(
            "Dummy RE task", labels=labels, templates=templates, valid_conditions=valid_conditions
        )

        features = [
            RelationClassificationFeatures(
                X="Billy Mays",
                Y="Tampa",
                inst_type="PERSON:CITY",
                context="Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
                label="per:city_of_death",
            ),
            RelationClassificationFeatures(
                X="Old Lane Partners",
                Y="Pandit",
                inst_type="ORGANIZATION:PERSON",
                context="Pandit worked at the brokerage Morgan Stanley for about 11 years until 2005, when he and some Morgan Stanley colleagues quit and later founded the hedge fund Old Lane Partners.",
                label="org:founded_by",
            ),
            RelationClassificationFeatures(
                X="He",
                Y="University of Maryland in College Park",
                inst_type="PERSON:ORGANIZATION",
                context="He received an undergraduate degree from Morgan State University in 1950 and applied for admission to graduate school at the University of Maryland in College Park.",
                label="no_relation",
            ),
        ]

        nlp = EntailmentClassifier(use_cuda=False, use_tqdm=False)

        nlp(
            task=task, 
            features=features,     
            return_labels=True,
            return_confidences=True,
            return_raw_output=True,
        )