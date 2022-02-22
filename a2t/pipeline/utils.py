from typing import Any


class PipelineElement:
    def __call__(self) -> Any:
        raise NotImplementedError(f"{self.__class__} is an abstract class that must be overrided.")
