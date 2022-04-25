"""The module `data` implements different dataloaders or `Dataset`s for predefined tasks.
"""
from .tacred import TACREDRelationClassificationDataset
from .babeldomains import BabelDomainsTopicClassificationDataset
from .wikievents import WikiEventsArgumentClassificationDataset
from .ace import ACEEventArgumentClassificationDataset
from .base import Dataset

PREDEFINED_DATASETS = {
    "tacred": TACREDRelationClassificationDataset,
    "babeldomains": BabelDomainsTopicClassificationDataset,
    "wikievents_arguments": WikiEventsArgumentClassificationDataset,
    "ace_arguments": ACEEventArgumentClassificationDataset,
}

__all__ = [
    "Dataset",
    "TACREDRelationClassificationDataset",
    "BabelDomainsTopicClassificationDataset",
    "WikiEventsArgumentClassificationDataset",
]

__pdoc__ = {
    "base": False,
    "babeldomains": False,
    "tacred": False,
    "wikievents": False,
}
