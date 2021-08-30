from .mnli import NLIRelationClassifierWithMappingHead
from .tacred import TACRED_LABELS, TACREDClassifier

__all__ = [
    'NLIRelationClassifierWithMappingHead',
    'TACREDClassifier',
    'TACRED_LABELS'
]