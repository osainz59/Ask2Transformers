from typing import Dict, List
from dataclasses import dataclass

from .base import BinaryTask, Features


@dataclass
class RelationClassificationFeatures(Features):
    X: str = None
    Y: str = None


class RelationClassificationTask(BinaryTask):
    """A class handler for Relation Classification task. It inherits from `BinaryTask` class.

    TODO: Add documentation.

    """

    def __init__(
        self,
        name: str,
        labels: List[str],
        *args,
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
            name (str): [description]
            labels (List[str]): [description]
            required_variables (List[str], optional): [description]. Defaults to ["X", "Y"].
            additional_variables (List[str], optional): [description]. Defaults to ["inst_type"].
            templates (Dict[str, List[str]], optional): [description]. Defaults to None.
            valid_conditions (Dict[str, List[str]], optional): [description]. Defaults to None.
            features_class (type, optional): [description]. Defaults to RelationClassificationFeatures.
            multi_label (bool, optional): [description]. Defaults to True.
            negative_label_id (int, optional): [description]. Defaults to 0.
        """
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


@dataclass
class TACREDFeatures(Features):
    subj: str = None
    obj: str = None


class TACREDRelationClassificationTask(RelationClassificationTask):
    def __init__(
        self, labels: List[str], templates: Dict[str, List[str]], valid_conditions: Dict[str, List[str]], **kwargs
    ) -> None:
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
