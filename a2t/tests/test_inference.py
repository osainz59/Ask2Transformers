import unittest

from a2t.base import EntailmentClassifier
from a2t.tasks.text_classification import TopicClassificationFeatures, TopicClassificationTask

import os

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
