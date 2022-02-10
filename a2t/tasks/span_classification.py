from typing import Dict, List
from dataclasses import dataclass

from .text_classification import IncorrectHypothesisTemplateError
from .base import UnaryTask, UnaryFeatures


@dataclass
class NamedEntityClassificationFeatures(UnaryFeatures):
    """A class handler for the Named Entity Classification features. It inherits from `UnaryFeatures`."""


class NamedEntityClassificationTask(UnaryTask):
    """A class handler for Named Entity Classification task. It inherits from `UnaryTask` class."""

    def __init__(
        self,
        name: str,
        labels: List[str],
        *args,
        required_variables: List[str] = ["X"],
        additional_variables: List[str] = ["inst_type"],
        templates: Dict[str, List[str]] = None,
        valid_conditions: Dict[str, List[str]] = None,
        hypothesis_template: str = "{X} is a {label}.",
        features_class: type = NamedEntityClassificationFeatures,
        multi_label: bool = True,
        negative_label_id: int = 0,
        **kwargs
    ) -> None:
        """Initialization of a NamedEntityClassificationTask task.

        Args:
            name (str): A name for the task that may be used for to differentiate task when saving.
            labels (List[str]): The labels for the task.
            required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `NamedEntityClassificationFeatures` class. Defaults to `["X", "Y"]`.
            additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `NamedEntityClassificationFeatures` class. Defaults to ["inst_type"].
            templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to None.
            valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
            hypothesis_template (str, optional): A meta template to generate hypothesis templates, if `templates` is None,
                then the templates will be the combinations of the `hypothesis_template` and the `labels`. It must contain the
                '{label}' placeholder. Defaults to "{X} is a {label}.".
            features_class (type, optional): The `Features` class related to the task. Defaults to NamedEntityClassificationFeatures.
            multi_label (bool, optional): Whether the task must be treated as multi-label or not. Defaults to True.
            negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to 0.

        Raises:
            IncorrectHypothesisTemplateError: Raised when the `hypotesis_template` argument does not contain the '{label}' placeholder.
        """

        if not templates:
            if "{label}" not in hypothesis_template:
                raise IncorrectHypothesisTemplateError(
                    "The hypothesis_template argument must contain the '{label}' placeholder."
                )

            templates = {
                label: [hypothesis_template.replace("{label}", label)]
                for i, label in enumerate(labels)
                if i != negative_label_id
            }

        super().__init__(
            *args,
            name=name,
            required_variables=required_variables,
            additional_variables=additional_variables,
            labels=labels,
            templates=templates,
            valid_conditions=valid_conditions,
            features_class=features_class,
            multi_label=multi_label,
            negative_label_id=negative_label_id,
            **kwargs
        )
