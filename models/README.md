# Models
Package based on the `modelers.Modeler` interface.
A `Modeler` maps a given `Dataset` to a novel one through some transformation.

The package offers standard modelers to preprocess datasets for the model of choice, namely:
- `ContinuousTreeModeler` encodes categorical variables with target encoding, and binary variables to `{0, 1}`.
- `OneHotTreeModeler` encodes categorical variables with one-hot encoding, and binary variables to `{0, 1}`.
- `SBRLModeler` encodes all variables in bins, then in one-hot.
- `CorelsModeler` as `SBRLModeler`.

`Modeler`s implement the `process` function:
```python
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
```
Conventionally, `dataset` **is deep-copied before any processing takes place**.