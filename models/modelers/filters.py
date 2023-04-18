from dataset import Dataset
from models import Modeler


class ContinuousModeler(Modeler):
    """Filter out all non-continuous features."""
    def __init__(self):
        super().__init__()

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        features_to_ignore = kwargs.get("features_to_ignore", [dataset.target_feature])
        filtered_out_features = [f for f in dataset.columns if f not in features_to_ignore
                                 and "int" not in dataset.dtypes[f].name or "float" not in dataset.dtypes[f].name]

        processed_dataset = dataset[[f for f in dataset.columns
                                     if f not in features_to_ignore and f not in filtered_out_features]]

        return processed_dataset
