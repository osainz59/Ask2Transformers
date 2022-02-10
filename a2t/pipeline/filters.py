from dataclasses import dataclass
from typing import List

from a2t.tasks.base import Features


class Filter:
    def __init__(self) -> None:
        raise NotImplementedError("This is an abstract class and cannot be intantiated.")

    def __call__(self) -> None:
        raise NotImplementedError("__call__ method must be implemented.")


class NegativeInstanceFilter(Filter):
    def __init__(self, negative_label: str = "O") -> None:
        self.negative_label = negative_label

    def __call__(self, features: List[Features]) -> List[Features]:
        return [feature for feature in features if feature.label != self.negative_label]
