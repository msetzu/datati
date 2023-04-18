"""Model the given dataset for the desired model of choice by transforming its features."""

from abc import abstractmethod, ABC

from dataset import Dataset


class Modeler(ABC):
    """Process the given dataset for the model of choice."""
    def __init__(self):
        pass

    @abstractmethod
    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
            dataset: The dataset to process.
            **kwargs: Keyword arguments.

        Returns:
            The processed dataset.
        """
        pass
