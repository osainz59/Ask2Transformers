from collections import defaultdict
import inspect
import warnings
from dataclasses import dataclass, field, fields
from typing import Callable, List, Dict, Union
import re
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from a2t.utils import find_optimal_threshold, apply_threshold


@dataclass
class Features:
    """A simple class to handle the features information.

    Args:
        context (str): The context sentence.
        label (str, optional): The label of the instance.
        inst_type (str, optional): The type of the instance. This information is used for the `valid_conditions' constraints.
    """

    context: str
    label: str = None
    inst_type: str = None


class IncorrectFeatureTypeError(Exception):
    pass


@dataclass
class Task:
    """Abstract class for Tasks definition.

    The method `_assert_constraints()` must be overrided.

    Args:
        name (str, optional): A name for the task that may be used for to differentiate task when saving. Defaults to None.
        required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `Features` class. Defaults to empty list.
        additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `Features` class. Defaults to empty list.
        labels (List[str], optional): The labels for the task. Defaults to empty list.
        templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to empty dict.
        valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
        negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to -1.
        multi_label (bool, optional): Whether the task must be treated as multi-label or not. You should treat as multi-label task a task that contains a negative label. Defaults to False.
        features_class (type, optional): The `Features` class related to the task. Default to `Features`.
    """

    name: str = None
    required_variables: List[str] = field(default_factory=list)
    additional_variables: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    templates: Dict[str, List[str]] = field(default_factory=dict)
    valid_conditions: Dict[str, List[str]] = None
    negative_label_id: int = -1  # -1 for no negative class
    multi_label: bool = False
    features_class: type = Features

    def __post_init__(self):
        self._assert_minimal_constraints()
        self._assert_constraints()

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)

        if not self.templates:
            self.templates = {}

        # Create the templates to label mapping
        self.template2label = defaultdict(list)
        for label, templates in self.templates.items():
            for template in templates:
                self.template2label[template].append(label)

        self.template_list = list(self.template2label.keys())
        template2id = {template: i for i, template in enumerate(self.template_list)}

        self.label2templateid = defaultdict(list)
        for label, templates in self.templates.items():
            self.label2templateid[label].extend([template2id[template] for template in templates])

        # Create the valid_conditions matrix
        if self.valid_conditions:
            self._valid_conditions = {}
            self._always_valid_labels = np.zeros(self.n_labels)
            self._always_valid_labels[self.negative_label_id] = 1.0
            for label, conditions in self.valid_conditions.items():
                if label not in self.labels:
                    continue
                for condition in conditions:
                    if condition == "*":
                        self._always_valid_labels[self.label2id[label]] = 1.0
                        continue
                    if condition not in self._valid_conditions:
                        self._valid_conditions[condition] = np.zeros(self.n_labels)
                        if self.negative_label_id >= 0:
                            self._valid_conditions[condition][self.negative_label_id] = 1.0
                    self._valid_conditions[condition][self.label2id[label]] = 1.0
        else:
            self._valid_conditions = None

        def idx2label(idx):
            return self.labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def __repr__(self) -> str:
        class_name = self.name if self.name else str(self.__class__)
        labels_repr = self.labels.__repr__()
        if len(labels_repr) > 89:
            labels_repr = self.labels[:3].__repr__().replace("]", ", ...]")
        templates_repr = len(self.template2label)
        feature_class_repr = str(self.features_class)

        return (
            f"{class_name} ("
            f"\n\tLabels: {labels_repr}"
            f"\n\tTemplates: {templates_repr}"
            f"\n\tFeatures: {feature_class_repr}"
            "\n)"
        )

    def _assert_constraints(self):
        raise NotImplementedError(f"{self.__class__} is an abstract class. This method should be implemented.")

    def _assert_minimal_constraints(self):
        assert len(self.labels) > 0, "The number of labels should be greather than 0."

        assert self.negative_label_id < len(
            self.labels
        ), "The id for the negative label should be lower than the amount of labels."

        if self.negative_label_id >= 0:
            assert self.templates is not None and len(
                [value for values in self.templates.values() for value in values]
            ), "`templates` parameter must not be None nor empty."

        # assert all(
        #     key in self.labels for key in self.templates.keys()
        # ), "All the keys of templates dicts must be defined on labels."
        for key in list(self.templates.keys()):
            if key not in self.labels:
                warnings.warn(f"Label {key} not found among valid labels. Templates for label {key} not loaded.")
                del self.templates[key]

        if self.valid_conditions:
            # assert all(
            #     key in self.labels for key in self.valid_conditions.keys()
            # ), "All the keys of valid_conditions dict must be defined on labels."
            for key in list(self.valid_conditions.keys()):
                if key not in self.labels:
                    warnings.warn(f"Label {key} not found among valid labels. Valid conditions for label {key} not loaded.")
                    del self.valid_conditions[key]

        assert all(
            var in self.features_class.__dict__["__dataclass_fields__"]
            for var in self.required_variables + self.additional_variables
        ), "All variables should be defined on the features_class."

        assert all(
            var.strip("{").strip("}") in [*self.required_variables, *self.additional_variables]
            for templates in self.templates.values()
            for template in templates
            for var in re.findall(r"{\w+}", template)
        )

    def assert_features_class(self, features: List[Features]) -> None:
        """Assert that all features are instance of the task specific `Features` class.

        Args:
            features (List[Features]): The list of features to check.

        Raises:
            IncorrectFeatureTypeError: Raised when any feature is not an instance of the task specific `Features` class.
        """
        for feature in features:
            if not isinstance(feature, self.features_class):
                raise IncorrectFeatureTypeError(
                    f"Incorrect feature type given. Expected {self.features_class} but obtained {type(feature)}."
                )

    def generate_premise_hypotheses_pairs(self, features: List[Features], sep_token: str = "</s>") -> List[str]:
        """Generate premise-hypothesis pairs based on the `Task` templates.

        Args:
            features (List[Features]): The list of features.
            sep_token (str, optional): The model specific separator token. Defaults to "</s>".

        Returns:
            List[str]: The list of premise-hypothesis pairs generated from the features and templates.
        """
        if not isinstance(features, list):
            features = [features]

        sentence_pairs = [
            f"{feature.context} {sep_token} {template.format(**feature.__dict__)}"
            for feature in features
            for template in self.template_list
        ]
        return sentence_pairs

    def reverse_to_labels(self, template_probs: np.ndarray, collate_fn: Callable = np.max) -> np.ndarray:
        """A function that maps template probabilities to label probabilites. By default, the maximum probabilities among
        label related templates is used.

        Args:
            template_probs (np.ndarray): (batch_size, n_templates) The templates probabilites.
            collate_fn (Callable, optional): The probabilites collate function. Defaults to np.max.

        Returns:
            np.ndarray: (batch_size, n_labels) The labels probabilities.
        """
        outputs = np.hstack(
            [
                collate_fn(template_probs[:, self.label2templateid[label]], axis=-1, keepdims=True)
                if label in self.label2templateid
                else np.zeros((template_probs.shape[0], 1))
                for label in self.labels
            ]
        )
        return outputs

    def apply_valid_conditions(self, features: List[Features], probs: np.ndarray) -> np.ndarray:
        """Applies the valid conditions to the labels probabilities. If a constraint is not satisfied the probability is set to 0.

        Args:
            features (List[Features]): (batch_size,) The list of features.
            probs (np.ndarray): (batch_size, n_labels) The labels probabilities.

        Returns:
            np.ndarray: (batch_size, n_labels) The labels probabilities.
        """
        if self._valid_conditions:
            mask_matrix = np.stack(
                [self._valid_conditions.get(feature.inst_type, np.zeros(self.n_labels)) for feature in features],
                axis=0,
            )
            probs = probs * np.logical_or(mask_matrix, self._always_valid_labels)  # TODO: Need a test
        return probs

    def compute_metrics(
        self, labels: np.ndarray, output: np.ndarray, threshold: Union[str, float] = "optimize"
    ) -> Dict[str, float]:
        """Compute the metrics for the given task. This method is abstract and needs to be overrided.

        Args:
            labels (np.ndarray): (batch_size,) The correct labels.
            output (np.ndarray): (batch_size, n_labels) The labels probabilities.
            threshold (Union[str, float], optional): The threshold to use on the evaluation. Options:

                * **"default"**: The threshold is set to 0.5.
                * **"optimize"**: Optimize the threshold with the `labels`. Intended to be used on the development split.
                * **`float`**: A specific float value for the threshold.

                Defaults to "optimize".

        Raises:
            NotImplementedError: Raise if not overrided.

        Returns:
            Dict[str, float]: Dict with the resulting metrics.
        """
        # TODO: Unittest
        raise NotImplementedError("This method must be implemented.")

    @classmethod
    def from_config(cls, file_path: str) -> object:
        """Loads the Task instance from a configuration file.

        Args:
            file_path (str): The path to the configuration file.

        Returns:
            Task: A `Task` instance based on the configuration file.
        """
        with open(file_path, "rt") as f:
            config = json.load(f)

        if "features_class" in config:
            components = config["features_class"].split(".")
            mod = __import__(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)

            config["features_class"] = mod

        params = set([p.name for p in fields(cls)]) | set(inspect.signature(cls).parameters.keys())
        params = {key: config[key] for key in params if key in config}

        return cls(**params)

    def to_config(self, file_path: str) -> None:
        """Saves the task instance to a configuration file.

        Args:
            file_path (str): The path to the configuration file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wt") as f:
            values = {key: value for key, value in vars(self).items()}
            values["features_class"] = values["features_class"].__module__ + "." + values["features_class"].__name__
            for key in ["label2id", "idx2label", "n_labels", "template2label", "label2templateid", "_valid_conditions"]:
                del values[key]

            json.dump(values, f, indent=4)


class ZeroaryFeatures(Features):
    """A features class for `ZeroaryTask`. It only requires a `context` argument."""

    pass


@dataclass
class ZeroaryTask(Task):
    """A `Task` implementation for Text Classification like tasks.

    Args:
        name (str, optional): A name for the task that may be used for to differentiate task when saving. Defaults to None.
        required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `ZeroaryFeatures` class. Defaults to empty list.
        additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `ZeroaryFeatures` class. Defaults to empty list.
        labels (List[str], optional): The labels for the task. Defaults to empty list.
        templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to empty dict.
        valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
        multi_label (bool, optional): Whether the task must be treated as multi-label or not. You should treat as multi-label task a task that contains a negative label. Defaults to False.
        features_class (type, optional): The `Features` class related to the task. Defaults to `ZeroaryFeatures`.
        negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to -1.
    """

    features_class: type = ZeroaryFeatures

    def _assert_constraints(self):
        # Assert the number of required variables to be 0
        assert len(self.required_variables) == 0, "Zero-ary tasks like Text classifiation do not require any variable."

    def compute_metrics(self, labels: np.ndarray, output: np.ndarray, threshold: Union[str, float] = None) -> Dict[str, float]:
        """Compute the metrics for the given task. By default on `ZeroaryTask` the Accuracy is computed.

        Args:
            labels (np.ndarray): (batch_size,) The correct labels.
            output (np.ndarray): (batch_size, n_labels) The labels probabilities.
            threshold (Union[str, float], optional): No threshold is needed on `ZeroaryTask`.

        Returns:
            Dict[str, float]: Dict with the resulting metrics.
        """
        # TODO: Unittest
        if threshold:
            warnings.warn(f"{self.__class__} do not require 'threshold', ignored.")

        return {"accuracy_score": accuracy_score(labels, output.argmax(-1))}


@dataclass
class UnaryFeatures(Features):
    """A features class for `UnaryTask`. It requires `context` and `X` arguments."""

    X: str = None


@dataclass
class UnaryTask(Task):
    """A `Task` implementation for Span Classification like tasks.

    Args:
        name (str, optional): A name for the task that may be used for to differentiate task when saving. Defaults to None.
        required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `UnaryFeatures` class. Defaults `["X"]`.
        additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `UnaryFeatures` class. Defaults to empty list.
        labels (List[str], optional): The labels for the task. Defaults to empty list.
        templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to empty dict.
        valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
        multi_label (bool, optional): Whether the task must be treated as multi-label or not. You should treat as multi-label task a task that contains a negative label. Defaults to False.
        features_class (type, optional): The `Features` class related to the task. Default to `UnaryFeatures`.
        negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to -1.
    """

    required_variables: List[str] = field(default_factory=lambda: ["X"])
    features_class: type = UnaryFeatures

    def _assert_constraints(self):
        # Assert the number of required variables to be 1
        assert len(self.required_variables) == 1, "Unary-ary tasks like Span classifiation requires 1 variable."

    def compute_metrics(self, labels: np.ndarray, output: np.ndarray, threshold: Union[str, float] = "optimize"):
        """Compute the metrics for the given task. By default on `UnaryTask` the Accuracy is computed if
        the `negative_label_id` is < 0, otherwise the Precision, Recall, F1-Score and positive Accuracy are
        computed.

        Args:
            labels (np.ndarray): (batch_size,) The correct labels.
            output (np.ndarray): (batch_size, n_labels) The labels probabilities.
            threshold (Union[str, float], optional): The threshold to use on the evaluation. Options:

                * "default": The threshold is set to 0.5.
                * "optimize": Optimize the threshold with the `labels`. Intended to be used on the development split.
                * `float`: A specific float value for the threshold.

                Defaults to "optimize".

        Returns:
            Dict[str, float]: Dict with the resulting metrics.
        """
        # TODO: Unittest
        if threshold not in ["default", "optimize"] and not isinstance(threshold, float):
            raise ValueError("Threshold must be either 'default', 'optimize' or a float value.")

        if threshold == "default":
            threshold = 0.5

        if threshold == "optimize":
            threshold, _ = find_optimal_threshold(labels, output, negative_label_id=self.negative_label_id)

        results = {"optimal_threshold": threshold}
        if self.negative_label_id < 0:
            results["accuracy_score"] = accuracy_score(labels, output.argmax(-1))
        else:
            output_ = apply_threshold(output, threshold=threshold, negative_label_id=self.negative_label_id)
            positive_labels = list(set(range(len(self.labels))) - set([self.negative_label_id]))
            output_pos = output.copy()
            output_pos[:, self.negative_label_id] = 0.0

            results["positive_accuracy"] = accuracy_score(
                labels[labels != self.negative_label_id], output_pos[labels != self.negative_label_id, :].argmax(-1)
            )

            pre, rec, f1, _ = precision_recall_fscore_support(labels, output_, labels=positive_labels, average="micro")
            results["precision"] = pre
            results["recall"] = rec
            results["f1-score"] = f1

        return results


@dataclass
class BinaryFeatures(Features):
    """A features class for `BinaryTask`. It requires `context`, `X` and `Y` arguments."""

    X: str = None
    Y: str = None


@dataclass
class BinaryTask(Task):
    """A `Task` implementation for Relation Classification like tasks.

    Args:
        name (str, optional): A name for the task that may be used for to differentiate task when saving. Defaults to None.
        required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `BinaryFeatures` class. Defaults `["X", "Y"]`.
        additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `BinaryFeatures` class. Defaults to empty list.
        labels (List[str], optional): The labels for the task. Defaults to empty list.
        templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to empty dict.
        valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
        multi_label (bool, optional): Whether the task must be treated as multi-label or not. You should treat as multi-label task a task that contains a negative label. Defaults to False.
        features_class (type, optional): The `Features` class related to the task. Default to `BinaryFeatures`.
        negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to -1.
    """

    required_variables: List[str] = field(default_factory=lambda: ["X", "Y"])
    features_class: type = BinaryFeatures

    def _assert_constraints(self):
        # Assert the number of required variables to be 2
        assert len(self.required_variables) == 2, "Binary-ary tasks like Tuple classifiation require 2 variable."

    def compute_metrics(self, labels: np.ndarray, output: np.ndarray, threshold: Union[str, float] = "optimize"):
        """Compute the metrics for the given task. By default on `BinaryTask` the Accuracy is computed if
        the `negative_label_id` is < 0, otherwise the Precision, Recall, F1-Score and positive Accuracy are
        computed.

        Args:
            labels (np.ndarray): (batch_size,) The correct labels.
            output (np.ndarray): (batch_size, n_labels) The labels probabilities.
            threshold (Union[str, float], optional): The threshold to use on the evaluation. Options:

                * "default": The threshold is set to 0.5.
                * "optimize": Optimize the threshold with the `labels`. Intended to be used on the development split.
                * `float`: A specific float value for the threshold.

                Defaults to "optimize".

        Returns:
            Dict[str, float]: Dict with the resulting metrics.
        """

        # TODO: Unittest + documentation
        if threshold not in ["default", "optimize"] and not isinstance(threshold, float):
            raise ValueError("Threshold must be either 'default', 'optimize' or a float value.")

        if threshold == "default":
            threshold = 0.5

        if threshold == "optimize":
            threshold, _ = find_optimal_threshold(labels, output, negative_label_id=self.negative_label_id)

        results = {"optimal_threshold": threshold}
        if self.negative_label_id < 0:
            results["accuracy_score"] = accuracy_score(labels, output.argmax(-1))
        else:
            output_ = apply_threshold(output, threshold=threshold, negative_label_id=self.negative_label_id)
            positive_labels = list(set(range(len(self.labels))) - set([self.negative_label_id]))
            output_pos = output.copy()
            output_pos[:, self.negative_label_id] = 0.0

            results["positive_accuracy"] = accuracy_score(
                labels[labels != self.negative_label_id], output_pos[labels != self.negative_label_id, :].argmax(-1)
            )

            pre, rec, f1, _ = precision_recall_fscore_support(labels, output_, labels=positive_labels, average="micro")
            results["precision"] = pre
            results["recall"] = rec
            results["f1-score"] = f1

        return results
