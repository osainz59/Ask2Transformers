from a2t.base import EntailmentClassifier
from a2t.pipeline.filters import Filter
from a2t.tasks import Task, Features
from a2t.pipeline.candidates import CandidateGenerator

from typing import Dict, List, Tuple, Union


class WronglyDefinedPipelineException(Exception):
    """Raised when the input-output of the pipeline is not well defined."""


class Pipeline(list):
    def __init__(
        self,
        *elements: List[Tuple[Union[CandidateGenerator, Task], str, str]],
        model: Union[str, EntailmentClassifier] = "roberta-large-mnli",
        threshold: float = 0.5,
        **kwargs,
    ) -> None:

        # Check whether the pipeline is well defined or not
        result_dict = ["initial_features"]
        for _, input_name, output_name in elements:
            if input_name not in result_dict:
                raise WronglyDefinedPipelineException(f"Input features {input_name} referenced before assignment.")
            result_dict.append(output_name)

        if not isinstance(model, EntailmentClassifier):
            model = EntailmentClassifier(model, **kwargs)

        self.model = model
        self.threshold = threshold
        self.config = kwargs

        super().__init__(elements)

    def __call__(self, input_features: List[Features]) -> Dict[str, List[Features]]:
        result_dict = {"initial_features": input_features}

        for elem, input_name, output_name in self:
            features = result_dict[input_name]
            if isinstance(elem, CandidateGenerator):
                features = elem(features)
            if isinstance(elem, Filter):
                features = elem(features)
            if isinstance(elem, Task):
                nlp = self.config.get(f"{elem.name}_model", self.model)
                if isinstance(nlp, str):
                    nlp = EntailmentClassifier(nlp, **self.config)
                threshold = self.config.get(f"{elem.name}_negative_threshold", self.threshold)
                predictions = nlp(task=elem, features=features, negative_threshold=threshold, return_labels=True)

                for feature, label in zip(features, predictions):
                    feature.label = label

            # Add the predictions to the output dict
            result_dict[output_name] = features

        result_dict.pop("initial_features")

        return result_dict
