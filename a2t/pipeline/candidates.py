from typing import Iterable, List, Union
from itertools import product
from collections import defaultdict

from a2t.tasks import Features, UnaryFeatures, BinaryFeatures


class CandidateGenerator:
    """A candidate generator. Maps `Features` from one task to another."""

    def get_input_features_class(self) -> Union[type, None]:
        raise NotImplementedError("The method `get_input_features_class()` must be overrided.")

    def get_output_features_class(self) -> type:
        raise NotImplementedError("The method `get_output_features_class()` must be overrided.")

    def __call__(self, input_features: List[Features]) -> Iterable[Features]:
        raise NotImplementedError("The method `__call__()` must be overrided.")

    @staticmethod
    def group_features(features: List[Features], by: str = "context") -> Iterable[List[Features]]:
        """Groups features by some specific attribute. This function can be used to group features that shares
        the same instance or sentence.

        Args:
            features (List[Features]): The complete list of features.
            by (str, optional): The attribute by which the features are grouped. Defaults to "context".

        Returns:
            Iterable[List[Features]]: An iterable of the features groups.
        """
        assert all(hasattr(feature, by) for feature in features)

        group_dict = defaultdict(list)
        for feature in features:
            group_dict[getattr(feature, by)].append(feature)

        return group_dict.values()


class UnaryToBinaryCandidateGenerator(CandidateGenerator):
    """A BinaryFeatures generator from UnaryFeatures.

    The model generates all possible combinations given a set of `UnaryFeatures`. Only candidates
    that has already the label assigned and share the same context are considered.
    """

    def __init__(self, unary_feature_class: type = UnaryFeatures, binary_feature_class: type = BinaryFeatures) -> None:
        """Initialization of `UnaryToBinaryCandidateGenerator` class.

        Args:
            unary_feature_class (type, optional): The input features class, must inherit from `UnaryFeatures`. Defaults to UnaryFeatures.
            binary_feature_class (type, optional): The output features class, must inherit from `UnaryFeatures`. Defaults to BinaryFeatures.
        """
        super().__init__()

        assert issubclass(unary_feature_class, UnaryFeatures) and issubclass(binary_feature_class, BinaryFeatures)

        self._input_features_class = unary_feature_class
        self._output_features_class = binary_feature_class

    def get_input_features_class(self) -> Union[type, None]:
        return self._input_features_class

    def get_output_features_class(self) -> type:
        return self._output_features_class

    def __call__(self, input_features: List[UnaryFeatures]) -> List[BinaryFeatures]:
        candidates = [
            features
            for features in input_features
            if isinstance(features, self.get_input_features_class()) and features.label is not None
        ]

        return_list = []
        for group in self.group_features(candidates, by="context"):
            for c1, c2 in product(group, group):
                if c1 != c2 and c1.context == c2.context:
                    return_list.append(
                        self.get_output_features_class()(
                            context=c1.context, X=c1.X, Y=c2.X, inst_type=f"{c1.label}:{c2.label}", label=None
                        )
                    )

        return return_list
