from typing import List

import numpy as np


class Dataset(list):
    """A simple class to handle the datasets.

    Inherits from `list`, so the instances should be added with `append` or `extend` methods to itself.
    """

    def __init__(self, labels: List[str], *args, **kwargs) -> None:
        """
        Args:
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__()

        self.labels2id = {label: i for i, label in enumerate(labels)}
        self.id2labels = {i: label for i, label in enumerate(labels)}

    @property
    def labels(self):
        # TODO: Unittest
        if not hasattr(self, "_labels"):
            self._labels = np.asarray([self.labels2id[inst.label] for inst in self])
        return self._labels
