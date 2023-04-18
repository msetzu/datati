import copy
from typing import List, Optional, Dict

import numpy
import pandas

from dataset import Dataset
from encoding.quantization import Quantiler
from models import Modeler
from encoding import Binner, NumericBinner


class TargetModeler(Modeler):
    """Map all categorical features with target encoding.
    Attributes:
        encoding_map: Dictionary category => encoding.
    """
    def __init__(self):
        super().__init__()
        self.encoding_map = dict()

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        dataset_copy = copy.deepcopy(dataset)
        nr_records = dataset_copy.shape[0]
        features_to_ignore = kwargs.get("features_to_ignore", list())
        target_feature = kwargs.get("target_feature", dataset_copy.target_feature)
        feature_name_separator = kwargs.get("feature_name_separator", "~TARGET_ENCODED_FOR~")
        categorical_features = [f for f in dataset_copy.columns
                                if dataset_copy.dtypes[f].name in ("object", "string", "category")
                                and f not in features_to_ignore]
        values_per_categorical_feature = list()
        # only use n - 1 encodings since the last one can be inferred from the other ones
        unique_target_values = dataset_copy[target_feature].unique()[:-1]

        for categorical_feature in categorical_features:
            feature_values = dataset_copy[categorical_feature]
            unique_feature_values = feature_values.unique()
            values_per_categorical_feature.append(unique_feature_values)
            target_counts_per_category = numpy.array(
                [numpy.array([dataset_copy[(dataset_copy[categorical_feature] == value)
                                             & (dataset_copy[target_feature] == target_value)].shape[0]
                              for target_value in unique_target_values])
                 for value in unique_feature_values])
            encoded_feature = (target_counts_per_category / nr_records)
            self.encoding_map[categorical_feature] = {feature_value: encoded_feature_value
                                                      for feature_value, encoded_feature_value in zip(unique_feature_values, encoded_feature)}

        for categorical_feature, unique_feature_values in zip(categorical_features, values_per_categorical_feature):
            # replace in the same position by splitting before and after columns, then joining them
            position = list(dataset_copy.columns).index(categorical_feature)
            # dataframe values must be hashable, hence need to first split numpy array, then create the encoded dataframe
            encoded_df = list()
            for target_index, target_value in enumerate(unique_target_values):
                # encoded_df = [self.encoding_map[categorical_feature][x] for x in dataset_copy[categorical_feature].values]
                encoded_feature = dataset_copy[categorical_feature].apply(lambda x:
                                                                          self.encoding_map[categorical_feature][x][target_index])
                encoded_df.append(pandas.DataFrame(encoded_feature))
            encoded_df = pandas.concat(encoded_df, axis="columns")
            encoded_df.columns = [f"{categorical_feature}{feature_name_separator}{target_value}"
                                  for target_value in unique_target_values]
            preceding_df = dataset_copy[list(dataset_copy.columns)[:position]]
            following_df = dataset_copy[list(dataset_copy.columns)[position + 1:]]
            dataset_copy = pandas.concat((preceding_df, encoded_df, following_df),
                                           axis="columns")
        dataset_copy = Dataset(dataset_copy)
        dataset_copy.copy_metadata_from(dataset_copy)
        dataset_copy.target_encoding_dictionary.update(self.encoding_map)

        return dataset_copy


class BinModeler(Modeler):
    """Transform the given dataset into bins."""
    def __init__(self, binners: Optional[Dict[str, Binner]] = None):
        super().__init__()
        self.binners = binners

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        dataset_copy = copy.deepcopy(dataset)
        features_to_ignore = kwargs.get("features_to_ignore", [dataset_copy.target_feature])
        bins_are_categories = kwargs.get("bins_are_categories", True)
        features = [f for f in dataset_copy.columns if f not in features_to_ignore]
        if self.binners is None:
            self.binners = {f: Quantiler(n_quantiles=4) for f in dataset_copy.columns}

        for feature in features:
            if isinstance(self.binners[feature], NumericBinner)\
                    and not (dataset_copy.dtypes[feature].name.startswith("int")
                             or dataset_copy.dtypes[feature].name.startswith("float")):
                continue

            if not self.binners[feature].is_fit:
                self.binners[feature] = self.binners[feature].fit(dataset_copy[feature])
            dataset_copy[feature] = dataset_copy[[feature]].applymap(lambda x: self.binners[feature].bin([x])[0])
            if bins_are_categories:
                dataset_copy = dataset_copy.astype({feature: "category"})

            dataset_copy.bins_encoding_dictionary.update({feature: self.binners[feature].encoding_map})

        return dataset_copy


class BoolToIntModeler(Modeler):
    def __init__(self, guess_booleans: bool = False):
        super().__init__()
        self.processed_features = set()
        self.guess = guess_booleans

    def guess_booleans(self, dataset: Dataset) -> List[str]:
        """Guess boolean variables in `dataset`.

        Args:
            dataset: The dataset.

        Returns:
            A list of guessed boolean features.
        """
        boolean_features = [f for f in dataset.columns
                            if len(set(dataset[f].unique())) == 2 and dataset.dtypes[f].name in ("string", "object", "bool")]

        return boolean_features

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        dataset_copy = copy.deepcopy(dataset)
        if self.guess_booleans:
            guessed_booleans = self.guess_booleans(dataset_copy)
            processed_dataset = dataset_copy.astype({f: "bool" for f in guessed_booleans})
        else:
            processed_dataset = dataset_copy

        for feature in processed_dataset.columns:
            if processed_dataset.dtypes[feature].name == "bool":
                processed_dataset = processed_dataset.astype({feature: "int8"})
                self.processed_features.add(feature)

        return processed_dataset


class IntToBoolModeler(Modeler):
    def __init__(self):
        super().__init__()
        self.processed_features = set()

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        dataset_copy = copy.deepcopy(dataset)
        features_to_convert = [f for f in dataset.columns
                               if dataset.dtypes[f].name.startswith("int") and f != dataset.target_feature]
        dataset_copy = dataset_copy.astype({feature: "bool" for feature in features_to_convert})

        return dataset_copy
