from a2t.base import EntailmentClassifier
from a2t.tasks import Task, Features
from a2t.pipeline.candidates import CandidateGenerator

from typing import List, Union


class Pipeline(list):
    def __init__(
        self,
        elements: List[Union[CandidateGenerator, Task]],
        model: Union[str, EntailmentClassifier] = "roberta-large-mnli",
        **kwargs
    ) -> None:

        if not isinstance(model, EntailmentClassifier):
            model = EntailmentClassifier(model, **kwargs)
        self.model = model

        super().__init__(elements)

    def __call__(self, input_features: List[Features]):
        pass
