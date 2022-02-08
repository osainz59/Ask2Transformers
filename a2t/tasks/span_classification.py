from typing import Dict, List
from dataclasses import dataclass

from .text_classification import IncorrectHypothesisTemplateError
from .base import UnaryTask, Features


@dataclass
class NamedEntityClassificationFeatures(Features):
    X: str = None


class NamedEntityClassificationTask(UnaryTask):
    """A class handler for Named Entity Classification task. It inherits from `UnaryTask` class.

    TODO: Add documentation.

    """

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
            name (str): [description]
            labels (List[str]): [description]
            required_variables (List[str], optional): [description]. Defaults to ["X"].
            additional_variables (List[str], optional): [description]. Defaults to ["inst_type"].
            templates (Dict[str, List[str]], optional): [description]. Defaults to None.
            valid_conditions (Dict[str, List[str]], optional): [description]. Defaults to None.
            hypothesis_template (str, optional): [description]. Defaults to "{X} is a {label}.".
            features_class (type, optional): [description]. Defaults to NamedEntityClassificationFeatures.
            multi_label (bool, optional): [description]. Defaults to True.
            negative_label_id (int, optional): [description]. Defaults to 0.

        Raises:
            IncorrectHypothesisTemplateError: [description]
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
