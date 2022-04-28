from typing import Dict, List
from dataclasses import dataclass

from .base import BinaryTask, BinaryFeatures, Features


@dataclass
class RelationClassificationFeatures(BinaryFeatures):
    """A class handler for the Relation Classification features. It inherits from `BinaryFeatures`."""


class RelationClassificationTask(BinaryTask):
    """A class handler for Relation Classification task. It inherits from `BinaryTask` class."""

    def __init__(
        self,
        name: str,
        labels: List[str],
        # *args,
        required_variables: List[str] = ["X", "Y"],
        additional_variables: List[str] = ["inst_type"],
        templates: Dict[str, List[str]] = None,
        valid_conditions: Dict[str, List[str]] = None,
        features_class: type = RelationClassificationFeatures,
        multi_label: bool = True,
        negative_label_id: int = 0,
        **kwargs
    ) -> None:
        """Initialization of a RelationClassificationTask task.

        Args:
            name (str): A name for the task that may be used for to differentiate task when saving.
            labels (List[str]): The labels for the task.
            required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `RelationClassificationFeatures` class. Defaults to `["X", "Y"]`.
            additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `RelationClassificationFeatures` class. Defaults to ["inst_type"].
            templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to None.
            valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
            features_class (type, optional): The `Features` class related to the task. Defaults to RelationClassificationFeatures.
            multi_label (bool, optional): Whether the task must be treated as multi-label or not. Defaults to True.
            negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to 0.
        """
        super().__init__(
            # *args,
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


@dataclass
class EventArgumentClassificationFeatures(Features):
    """A class handler for the Event Argument Classification features. It inherits from `BinaryFeatures`."""

    trg: str = None
    arg: str = None
    trg_type: str = None
    trg_subtype: str = None


class EventArgumentClassificationTask(BinaryTask):
    """A class handler for Event Argument Classification task. It inherits from `BinaryTask` class."""

    def __init__(
        self,
        name: str,
        labels: List[str],
        required_variables: List[str] = ["trg", "arg"],
        additional_variables: List[str] = ["inst_type", "trg_type", "trg_subtype"],
        templates: Dict[str, List[str]] = None,
        valid_conditions: Dict[str, List[str]] = None,
        features_class: type = EventArgumentClassificationFeatures,
        multi_label: bool = True,
        negative_label_id: int = 0,
        **kwargs
    ) -> None:
        """Initialization of a RelationClassificationTask task.

        Args:
            name (str): A name for the task that may be used for to differentiate task when saving.
            labels (List[str]): The labels for the task.
            required_variables (List[str], optional): The variables required to perform the task and must be implemented by the `EventArgumentClassificationFeatures` class. Defaults to `["trg", "arg"]`.
            additional_variables (List[str], optional): The variables not required to perform the task and must be implemented by the `EventArgumentClassificationFeatures` class. Defaults to ["inst_type", "trg_type", "trg_subtype"].
            templates (Dict[str, List[str]], optional): The templates/verbalizations for the task. Defaults to None.
            valid_conditions (Dict[str, List[str]], optional): The valid conditions or constraints for the task. Defaults to None.
            features_class (type, optional): The `Features` class related to the task. Defaults to EventArgumentClassificationFeatures.
            multi_label (bool, optional): Whether the task must be treated as multi-label or not. Defaults to True.
            negative_label_id (int, optional): The index of the negative label or -1 if no negative label exist. A negative label is for example the class `Other` on NER, that means that the specific token is not a named entity. Defaults to 0.
        """
        super().__init__(
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


@dataclass
class TACREDFeatures(Features):
    """A class handler for the TACRED features. It inherits from `Features`."""

    subj: str = None
    obj: str = None


class TACREDRelationClassificationTask(RelationClassificationTask):
    """A class handler for TACRED Relation Classification task. It inherits from `RelationClassificationTask` class."""

    def __init__(
        self, labels: List[str], templates: Dict[str, List[str]], valid_conditions: Dict[str, List[str]], **kwargs
    ) -> None:
        """Initialization of the TACRED RelationClassification task

        Args:
            labels (List[str]): The labels for the task.
            templates (Dict[str, List[str]]): The templates/verbalizations for the task.
            valid_conditions (Dict[str, List[str]]): The valid conditions or constraints for the task.
        """
        for key in ["name", "required_variables", "additional_variables", "features_class", "multi_label", "negative_label_id"]:
            kwargs.pop(key, None)
        super().__init__(
            "TACRED Relation Classification task",
            labels=labels,
            required_variables=["subj", "obj"],
            additional_variables=["inst_type"],
            templates=templates,
            valid_conditions=valid_conditions,
            features_class=TACREDFeatures,
            multi_label=True,
            negative_label_id=0,
            **kwargs
        )
