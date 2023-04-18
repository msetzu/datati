"""Model the given dataset in a binary format by mapping all features to a 0-1 domain."""
from typing import Optional, Dict

from dataset import Dataset
from encoding import Binner
from models.modelers.one_hot import BinaryBoolModeler


class CorelsModeler(BinaryBoolModeler):
    def __init__(self, binner: Optional[Binner | Dict[str, Binner]] = None):
        """
        Args:
            binner: Optional binner to bin features. Either a unique binner, which will be applied to all features,
                    or a dictionary feature => binner, in which case dictionary[feature] will be applied to feature
                    only.
        """
        super(CorelsModeler, self).__init__(binner)

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        binary_dataset = super(CorelsModeler, self).process(dataset, **kwargs)

        return binary_dataset.astype({f: "int8" for f in binary_dataset.columns})


class SBRLModeler(BinaryBoolModeler):
    def __init__(self, binner: Optional[Binner | Dict[str, Binner]] = None):
        """
        Args:
            binner: Optional binner to bin features. Either a unique binner, which will be applied to all features,
                    or a dictionary feature => binner, in which case dictionary[feature] will be applied to feature
                    only.
        """
        super(SBRLModeler, self).__init__(binner)

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        binary_dataset = super(SBRLModeler, self).process(dataset, **kwargs)

        return binary_dataset.astype({f: "int8" for f in binary_dataset.columns})
