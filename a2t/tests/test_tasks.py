import unittest
import tempfile
from dataclasses import dataclass
from a2t.base import EntailmentClassifier

from a2t.tasks.base import IncorrectFeatureTypeError, Task, Features, ZeroaryTask, UnaryTask, BinaryTask
from a2t.tasks.span_classification import NamedEntityClassificationFeatures, NamedEntityClassificationTask
from a2t.tasks.text_classification import (
    IncorrectHypothesisTemplateError,
    TopicClassificationTask,
    TopicClassificationFeatures,
)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TestTasks(unittest.TestCase):
    def test_assert_minimal_constraints(self):

        # Check number of labels greater than 0
        self.assertRaises(AssertionError, Task)

        # Check templates keys
        self.assertRaises(AssertionError, Task, labels=["O"], templates={"B": []})

        # Check valid conditions keys
        self.assertRaises(AssertionError, Task, labels=["O"], templates={"O": []}, valid_conditions={"B": []})

        # Check negative label id to be smaller than the number of labels
        self.assertRaises(AssertionError, Task, labels=["O"], templates={"O": []}, negative_label_id=1)

        # Check required and additional variables
        @dataclass
        class DummyFeatures(Features):
            X: str = None

        class DummyTask(Task):
            def _assert_constraints(self):
                pass

        _ = DummyTask(required_variables=["X"], labels=["O"], templates={"O": []}, features_class=DummyFeatures)

        self.assertRaises(AssertionError, DummyTask, required_variables=["X"], labels=["O"], templates={"O": []})

        _ = DummyTask(
            required_variables=["X"], labels=["O"], templates={"O": ["{X} is a Potato."]}, features_class=DummyFeatures
        )

        self.assertRaises(
            AssertionError, DummyTask, required_variables=["X"], labels=["O"], templates={"O": ["{Y} is a Potato."]}
        )

        # Abstract class should not be instantiated
        self.assertRaises(NotImplementedError, Task, labels=["O"])

    def test_assert_constraints(self):
        _ = ZeroaryTask(labels=["O"])

        @dataclass
        class UnaryFeatures(Features):
            X: str = None

        _ = UnaryTask(labels=["O"], required_variables=["X"], features_class=UnaryFeatures)

        @dataclass
        class BinaryFeatures(Features):
            X: str = None
            Y: str = None

        _ = BinaryTask(labels=["O"], required_variables=["X", "Y"], features_class=BinaryFeatures)

    def test_assert_feature_type(self):
        @dataclass
        class UnaryFeatures(Features):
            X: str = None

        unary_task = UnaryTask(labels=["O"], required_variables=["X"], features_class=UnaryFeatures)

        unary_features = [UnaryFeatures(context="", label="", X=str(i)) for i in range(10)]

        @dataclass
        class BinaryFeatures(Features):
            X: str = None
            Y: str = None

        binary_task = BinaryTask(labels=["O"], required_variables=["X", "Y"], features_class=BinaryFeatures)

        binary_features = [BinaryFeatures(context="", label="", X=str(i), Y=str(i + 10)) for i in range(10)]

        # Correct
        unary_task.assert_features_class(unary_features)
        binary_task.assert_features_class(binary_features)

        # Incorrect
        self.assertRaises(IncorrectFeatureTypeError, unary_task.assert_features_class, binary_features)
        self.assertRaises(IncorrectFeatureTypeError, binary_task.assert_features_class, unary_features)

    def test_save_and_load(self):
        task = ZeroaryTask(
            "Topic Classification task.",
            labels=["Medicine", "Sports", "Politics"],
            templates={
                "Medicine": ["Topic about health.", "Topic about medicine."],
                "Sports": ["Topic about footbal.", "Topic about sports."],
                "Politics": ["Topic about corruption.", "Topic about politics."],
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            task.to_config(os.path.join(tmpdir, "task.config.json"))
            task2 = ZeroaryTask.from_config(os.path.join(tmpdir, "task.config.json"))

        self.assertEqual(task, task2)


class TestTopicClassification(unittest.TestCase):
    def test_hypothesis_generation(self):
        labels = ["Culture", "Health", "Army"]

        features = [
            TopicClassificationFeatures(context="hospital: a health facility where patients receive treatment.", label="Health")
        ]

        hypothesis_template = "The domain of the sentence is {label}."

        task = TopicClassificationTask("Test Topic task", labels=labels, hypothesis_template=hypothesis_template)

        premise_hypothesis_pairs = task.generate_premise_hypotheses_pairs(features)
        hypotheses = [hypothesis_template.format(label=label) for label in labels]

        self.assertTrue(len(premise_hypothesis_pairs) == len(hypotheses), "The amount of hypothesis generated is not correct.")

        self.assertTrue(
            all(correct in generated for correct, generated in zip(hypotheses, premise_hypothesis_pairs)),
            "The generated hypothesis does not match the correct ones.",
        )

        self.assertRaises(
            IncorrectHypothesisTemplateError,
            TopicClassificationTask,
            "Test Topic task",
            labels=labels,
            hypothesis_template="This template does not contain a label placeholder.",
        )


class TestNamedEntityClassificationTask(unittest.TestCase):
    def test_hypothesis_generation(self):
        labels = ["O", "Person", "Location", "Organization"]

        valid_conditions = {"Person": ["*"], "Location": ["NNP"], "Organization": ["NNP"]}

        task = NamedEntityClassificationTask("Dummy NER task", labels=labels, valid_conditions=valid_conditions)

        features = [
            NamedEntityClassificationFeatures(
                context="Peter won an award in London.", label="Person", inst_type="NNP", X="Peter"
            ),
            NamedEntityClassificationFeatures(
                context="Peter won an award in London.", label="Location", inst_type="NNP", X="London"
            ),
        ]

        self.assertEqual(
            task.generate_premise_hypotheses_pairs(features),
            [
                "Peter won an award in London. </s> Peter is a Person.",
                "Peter won an award in London. </s> Peter is a Location.",
                "Peter won an award in London. </s> Peter is a Organization.",
                "Peter won an award in London. </s> London is a Person.",
                "Peter won an award in London. </s> London is a Location.",
                "Peter won an award in London. </s> London is a Organization.",
            ],
        )

    def test_inference(self):
        labels = ["O", "Person", "Location", "Organization"]

        valid_conditions = {"Person": ["NNP"], "Location": ["NNP"], "Organization": ["NNP"]}

        task = NamedEntityClassificationTask("Dummy NER task", labels=labels, valid_conditions=valid_conditions)

        features = [
            NamedEntityClassificationFeatures(
                context="Peter won an award in London.", label="Person", inst_type="NNP", X="Peter"
            ),
            NamedEntityClassificationFeatures(
                context="Peter won an award in London.", label="Location", inst_type="NNP", X="London"
            ),
        ]

        nlp = EntailmentClassifier(use_cuda=False, use_tqdm=False)

        preds = nlp(task=task, features=features, negative_threshold=0.5, return_labels=True, topk=1)

        self.assertEqual(preds, [feature.label for feature in features])
