"""The module `data` implements different dataloaders or `Dataset`s for predefined tasks.
"""
from .tacred import TACREDRelationClassificationDataset
from .babeldomains import BabelDomainsTopicClassificationDataset
from .base import Dataset

PREDEFINED_DATASETS = {"tacred": TACREDRelationClassificationDataset, "babeldomains": BabelDomainsTopicClassificationDataset}

__all__ = ["Dataset", "TACREDRelationClassificationDataset", "BabelDomainsTopicClassificationDataset"]
