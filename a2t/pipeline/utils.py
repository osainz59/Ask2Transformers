from typing import Any, List

from a2t.tasks import Features


class PipelineElement:
    """An abstract class to handle Pipeline custom elements.

    This class should be implemented and the `__call__` method overrided.
    """

    def __call__(self, input_features: List[Features]) -> List[Features]:
        """A method that will convert/filter/generate new list of `a2t.tasks.Features` from others.

        Args:
            input_features (List[Features]): The list of old `a2t.tasks.Features`.

        Raises:
            NotImplementedError: Raised when the method is not overrided from the child class.

        Returns:
            List[Features]: The new list of `a2t.tasks.Features`.
        """
        raise NotImplementedError(f"{self.__class__} is an abstract class that must be overrided.")
