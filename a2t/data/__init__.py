"""The module `data` implements different dataloaders or `Dataset`s for predefined tasks.
"""
from .tacred import TACREDRelationClassificationDataset
from .babeldomains import BabelDomainsTopicClassificationDataset
from .wikievents import WikiEventsArgumentClassificationDataset
from .base import Dataset

PREDEFINED_DATASETS = {
    "tacred": TACREDRelationClassificationDataset,
    "babeldomains": BabelDomainsTopicClassificationDataset,
    "wikievents_arguments": WikiEventsArgumentClassificationDataset,
}

__all__ = [
    "Dataset",
    "TACREDRelationClassificationDataset",
    "BabelDomainsTopicClassificationDataset",
    "WikiEventsArgumentClassificationDataset",
]
