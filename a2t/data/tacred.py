from typing import List
import json

from a2t.tasks.tuple_classification import TACREDFeatures
from .base import Dataset


class TACREDRelationClassificationDataset(Dataset):
    """A class to handle TACRED datasets.

    This class converts TACRED data files into a list of `a2t.tasks.TACREDFeatures`.
    """

    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        """
        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(labels=labels, *args, **kwargs)

        with open(input_path, "rt") as f:
            for i, line in enumerate(json.load(f)):
                self.append(
                    TACREDFeatures(
                        subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                        obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                        inst_type=f"{line['subj_type']}:{line['obj_type']}",
                        context=" ".join(line["token"])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                        label=line["relation"],
                    )
                )
