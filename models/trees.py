import copy

from dataset import Dataset
from models import Modeler
from models.modelers.numeric import TargetModeler, BoolToIntModeler
from models.modelers.one_hot import OneHotModeler
from pipeline import Pipeline


class ContinuousTreeModeler(Modeler):
    """Model data as continuous, mapping boolean features to 0/1, and categorical features to target encoders."""
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline(TargetModeler(), BoolToIntModeler(guess_booleans=True))

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        return self.pipeline(dataset, **kwargs)


class OneHotTreeModeler(Modeler):
    """Model data as continuous and discrete, mapping boolean features to 0/1, and categorical features to one-hot."""
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline(OneHotModeler(), BoolToIntModeler(guess_booleans=True))

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        dataset_copy = copy.deepcopy(dataset)

        return self.pipeline(dataset_copy)
