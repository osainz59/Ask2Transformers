from typing import Dict, List, Callable

from dataclasses import dataclass

from .base import ZeroaryTask, Features


class IncorrectHypothesisTemplateError(Exception):
    """Raised when the `hypotesis_template` argument does not contain the '{label}' placeholder."""

    pass


@dataclass
class TextClassificationFeatures(Features):
    """A class handler for the Text Classification features. It inherits from `Features`."""

    pass


class TextClassificationTask(ZeroaryTask):
    """A class handler for Text Classification tasks. It inherits from `ZeroaryTask` class."""

    def __init__(
        self,
        name: str = None,
        labels: List[str] = None,
        templates: Dict[str, List[str]] = None,
        hypothesis_template: str = "It was {label}.",
        features_class: type = TextClassificationFeatures,
        multi_label: bool = False,
        **kwargs,
    ):
        """_summary_

        Args:
            name (str, optional): A name for the task that may be used for to differentiate task when saving. Defaults to None.
            labels (List[str], optional): The labels for the task. Defaults to empty list.
            templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to None.
            hypothesis_template (str, optional): A meta template to generate hypothesis templates, if `templates` is None,
                then the templates will be the combinations of the `hypothesis_template` and the `labels`. It must contain the
                '{label}' placeholder. Defaults to "The domain of the sentence is about {label}.".
            features_class (type, optional): The `Features` class related to the task. Defaults to TextClassificationFeatures.
            multi_label (bool, optional): Whether the task must be treated as multi-label or not. Defaults to False.

        Raises:
            IncorrectHypothesisTemplateError: Raised when the `hypotesis_template` argument does not contain the '{label}' placeholder.
        """
        if not templates:

            if "{label}" not in hypothesis_template:
                raise IncorrectHypothesisTemplateError(
                    "The hypothesis_template argument must contain the '{label}' placeholder."
                )
            templates = {label: [hypothesis_template.format(label=label)] for label in labels}

        super().__init__(
            name=name,
            required_variables=[],
            additional_variables=[],
            labels=labels,
            templates=templates,
            valid_conditions=None,
            features_class=features_class,
            multi_label=multi_label,
            **kwargs,
        )


@dataclass
class TopicClassificationFeatures(Features):
    """A class handler for the Topic Classification features. It inherits from `Features`."""

    pass


class TopicClassificationTask(ZeroaryTask):
    """A class handler for Topic Classification task. It inherits from `ZeroaryTask` class."""

    def __init__(
        self,
        name: str = None,
        labels: List[str] = None,
        templates: Dict[str, List[str]] = None,
        hypothesis_template: str = "The domain of the sentence is about {label}.",
        features_class: type = TopicClassificationFeatures,
        preprocess_labels: bool = False,
        preprocess_fn: Callable = None,
        **kwargs,
    ) -> None:
        """Initialization of a TopicClassification task.

        Args:
            name (str, optional): A name for the task that may be used for to differentiate task when saving. Defaults to None.
            labels (List[str]): The labels for the task. Defaults to empty list.
            templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to None.
            hypothesis_template (str, optional): A meta template to generate hypothesis templates, if `templates` is None,
                then the templates will be the combinations of the `hypothesis_template` and the `labels`. It must contain the
                '{label}' placeholder. Defaults to "The domain of the sentence is about {label}.".
            features_class (type, optional): The `Features` class related to the task. Defaults to TopicClassificationFeatures.
            preprocess_labels (bool, optional): Whether to split the topic labels. Defaults to True.
            preprocess_fn (Callable, optional): The function that is applied if `split_labels` is True. If None then
                `TopicClassificationTask._split_labels_fn` is applied. Defaults to None.

        Raises:
            IncorrectHypothesisTemplateError: Raised when the `hypotesis_template` argument does not contain the '{label}' placeholder.
        """
        if not templates:

            if "{label}" not in hypothesis_template:
                raise IncorrectHypothesisTemplateError(
                    "The hypothesis_template argument must contain the '{label}' placeholder."
                )

            if preprocess_labels:
                split_labels_fn = preprocess_fn if preprocess_fn is not None else self._split_and_extend_labels_fn
                templates = {
                    label: [hypothesis_template.format(label=partial_label) for partial_label in split_labels_fn(label)]
                    for label in labels
                }
            else:
                templates = {label: [hypothesis_template.format(label=label)] for label in labels}

        super().__init__(
            name=name,
            required_variables=[],
            additional_variables=[],
            labels=labels,
            templates=templates,
            valid_conditions=None,
            features_class=features_class,
            **kwargs,
        )

    @staticmethod
    def _split_labels_fn(label: str) -> List[str]:
        labels = [
            partial_label.strip().capitalize()
            for partial_label in label.split(",")
            for partial_label in partial_label.split("and")
            if len(partial_label.strip())
        ]
        return list(set(labels))

    @staticmethod
    def _split_and_extend_labels_fn(label: str) -> List[str]:
        labels = [label] + [
            partial_label.strip().capitalize()
            for partial_label in label.split(",")
            for partial_label in partial_label.split("and")
            if len(partial_label.strip())
        ]
        return list(set(labels))
