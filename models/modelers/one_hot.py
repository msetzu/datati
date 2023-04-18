import copy
from functools import partial
from typing import Dict

import numpy
import pandas

from encoding import Binner

from dataset import Dataset
from models import Modeler
from models.modelers.numeric import BinModeler, IntToBoolModeler
from pipeline import Pipeline


class BinaryBoolModeler(Modeler):
    """Transform the given dataset into a binary one by:
        - binning its numerical features;
        - create a one-hot encoding of such bins.

    Attributes:
        one_hot_encoding_dictionary: Dictionary mapping each feature and value to the position in the one-hot space.
        bins_encoding_dictionary: Dictionary mapping each bin to its elements.
        base_binners: The base binner used to create a binner for each feature.
        fit_binners: Binners fit to each feature.
    """
    def __init__(self, binner: Binner | Dict[str, Binner]):
        """
        Args:
            binner: Base Binner. A copy of such binner will be fit on each feature to bin.
        """
        super().__init__()
        self.one_hot_encoding_dictionary = dict()
        self.bins_encoding_dictionary = dict()
        self.base_binners = binner
        self.fit_binners = dict()
        self.pipeline = Pipeline(BinModeler(self.base_binners), OneHotModeler(), IntToBoolModeler())

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
            dataset: The dataset to process.
            **kwargs: Keyword arguments.

        Returns:
            The processed dataset.
        """
        dataset_copy = copy.deepcopy(dataset)
        
        return self.pipeline(dataset_copy, **kwargs)


class OneHotModeler(Modeler):
    """Transform the given dataset into a numeric one by one-hot-encoding categorical features.

    Attributes:
        one_hot_encoding_dictionary: Dictionary mapping each feature and value to the position in the one-hot space.
    """
    def __init__(self):
        super().__init__()
        self.one_hot_encoding_dictionary = dict()

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
            dataset: The dataset to process.
            **kwargs: Keyword arguments.

        Returns:
            The processed dataset.
        """
        def one_hot_encode_value(zeros_base, values_indicators, v):
            encoded = zeros_base.copy()
            encoded[values_indicators[v]] = 1

            return encoded

        encoded_features = list()
        dataset_copy = copy.deepcopy(dataset)
        features_to_ignore = kwargs.get("features_to_ignore", [dataset_copy.target_feature])
        feature_name_separator = kwargs.get("feature_name_separator", "~ONE-HOT~")
        
        categorical_features = [f for f in dataset_copy.columns if dataset_copy.dtypes[f].name in ("object", "string", "category")
                                and f not in features_to_ignore]
        for f in categorical_features:
            self.one_hot_encoding_dictionary[f] = dict()
            values = dataset_copy[f]
            unique_values = sorted(values.unique())
            base = numpy.zeros(len(unique_values), ).astype(int)
            for encoded_position, value in enumerate(unique_values):
                self.one_hot_encoding_dictionary[f][value] = encoded_position

            # encoding_function = partial(one_hot_encode_value, base, self.one_hot_encoding_dictionary[f])
            # encoded_feature = values.apply(encoding_function)
            # encoded_feature = numpy.array([v for v in encoded_feature])
            encoding_function = partial(one_hot_encode_value, base, self.one_hot_encoding_dictionary[f])
            encoded_feature = numpy.array([encoding_function(v) for v in dataset_copy[f]])

            encoded_df = pandas.DataFrame(encoded_feature,
                                          columns=[f"{f}{feature_name_separator}{value}"
                                                   for value in unique_values]).astype(int)
            encoded_features.append(encoded_df)

        desired_column_list = list(dataset_copy.columns)
        for f, encoded_f in zip(categorical_features, encoded_features):
            position = desired_column_list.index(f)
            preceding_columns = desired_column_list[:position]
            following_columns = desired_column_list[position + 1:]
            desired_column_list = preceding_columns + list(encoded_f.columns) + following_columns
        dataset_copy = pandas.concat((dataset_copy, *encoded_features), axis="columns")
        dataset_copy = dataset_copy[desired_column_list]

        dataset_copy = Dataset(dataset_copy)
        dataset_copy.copy_metadata_from(dataset)
        dataset_copy.one_hot_encoding_dictionary.update(self.one_hot_encoding_dictionary)

        return dataset_copy
