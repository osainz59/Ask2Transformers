from pprint import pprint
import warnings
from a2t.base import EntailmentClassifier
from a2t.tasks import Task, Features
from .filters import Filter
from .candidates import CandidateGenerator
from .utils import PipelineElement

from typing import Dict, Iterable, List, Tuple, Union


class WronglyDefinedPipelineException(Exception):
    """Raised when the input-output of the pipeline is not well defined."""


class Pipeline(list):
    def __init__(
        self,
        *elements: List[Tuple[Union[CandidateGenerator, Task, Filter], str, str]],
        model: Union[str, EntailmentClassifier] = "roberta-large-mnli",
        threshold: float = 0.5,
        **kwargs,
    ) -> None:

        if not isinstance(model, EntailmentClassifier):
            model = EntailmentClassifier(model, **kwargs)

        self.model = model
        self.threshold = threshold
        self.config = kwargs

        super().__init__(self._verify_pipeline_elements(*elements))

    @staticmethod
    def _verify_pipeline_elements(*elements) -> Iterable[Union[Task, PipelineElement]]:
        result_dict = ["input_features"]
        for element in elements:
            if not (isinstance(element, tuple) and len(element) == 3):
                warnings.warn(
                    "{element} is not a tuple o (PipelineElement/Task, input_features, output_features) style. Ignored"
                )
                continue

            pipeline_elem, input_name, output_name = element
            if not (isinstance(pipeline_elem, Task) or isinstance(pipeline_elem, PipelineElement)):
                warnings.warn(f"{pipeline_elem} is not a PipelineElement nor Task. Ignored.")
                continue

            if input_name not in result_dict:
                raise WronglyDefinedPipelineException(f"Input features {input_name} referenced before assignment.")
            result_dict.append(output_name)

            yield element

    def __call__(self, input_features: List[Features]) -> Dict[str, List[Features]]:
        result_dict = {"input_features": input_features}

        for elem, input_name, output_name in self:
            if input_name is None:
                input_name = "input_features"
            features = result_dict[input_name]
            if isinstance(elem, PipelineElement):
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

        result_dict.pop("input_features")

        return result_dict
