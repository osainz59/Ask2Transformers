from typing import List

from a2t.tasks.text_classification import TopicClassificationFeatures
from .base import Dataset


class BabelDomainsTopicClassificationDataset(Dataset):
    """A class to handle BabelDomains datasets.

    This class converts BabelDomains data files into a list of `a2t.tasks.TopicClassificationFeatures`.
    """

    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        """
        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(labels=labels, *args, **kwargs)

        with open(input_path, "rt") as f:
            for line in f:
                _, label, context = line.strip().split("\t")
                self.append(TopicClassificationFeatures(context=context, label=label))
