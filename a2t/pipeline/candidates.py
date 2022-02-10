from typing import Iterable, List, Union
from itertools import product

from a2t.tasks import Features, UnaryFeatures, BinaryFeatures


class CandidateGenerator:
    """A candidate generator. Maps `Features` from one task to another."""

    def get_input_features_class(self) -> Union[type, None]:
        raise NotImplementedError("The method `get_input_features_class()` must be overrided.")

    def get_output_features_class(self) -> type:
        raise NotImplementedError("The method `get_output_features_class()` must be overrided.")

    def __call__(self, input_features: List[Features]) -> Iterable[Features]:
        raise NotImplementedError("The method `__call__()` must be overrided.")


class UnaryToBinaryCandidateGenerator(CandidateGenerator):
    """A BinaryFeatures generator from UnaryFeatures.

    The model generates all possible combinations given a set of `UnaryFeatures`. Only candidates
    that has already the label assigned and share the same context are considered.
    """

    def __init__(self, unary_feature_class: type = UnaryFeatures, binary_feature_class: type = BinaryFeatures) -> None:
        super().__init__()

        self._input_features_class = unary_feature_class
        self._output_features_class = binary_feature_class

    def get_input_features_class(self) -> Union[type, None]:
        return self._input_features_class

    def get_output_features_class(self) -> type:
        return self._output_features_class

    def __call__(self, input_features: List[UnaryFeatures]) -> Iterable[BinaryFeatures]:
        candidates = [
            features
            for features in input_features
            if isinstance(features, self.get_input_features_class()) and features.label is not None
        ]

        for c1, c2 in product(candidates, candidates):
            if c1 != c2 and c1.context == c2.context:
                yield self.get_output_features_class()(
                    context=c1.context, X=c1.X, Y=c2.X, inst_type=f"{c1.label}:{c2.label}", label=None
                )
