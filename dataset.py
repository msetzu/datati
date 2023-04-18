from __future__ import annotations
import warnings

from pandas import DataFrame, Series
from pandas import Index
from pandas._libs import lib

warnings.simplefilter(action="ignore", category=UserWarning)

import copy
from typing import Optional, Dict, TypeVar, Sequence, Tuple, Literal, Callable, Hashable, Mapping

import numpy
import pandas
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets import load_dataset as load_huggingface_dataset
from pandas._typing import IgnoreRaise, ArrayLike, Axes, Dtype, FillnaOptions, Axis, Level, IndexLabel, Suffixes
from pandas._typing import QuantileInterpolation, AnyArrayLike, Frequency

T = TypeVar("T")
S = TypeVar("S")


class Dataset(DataFrame):
    """Dataset class.

    Attributes:
        one_hot_encoding_dictionary: Encoding dictionary for one-hot variables.
        bins_encoding_dictionary: Encoding dictionary for binned variables.
        target_encoding_dictionary: Encoding dictionary for target variables.
        target_feature: The target feature of this dataset.
    """
    def __init__(self, dataset: Optional[ArrowDataset, numpy.ndarray, DataFrame, str] = None,
                 load_from: Optional[str] = None, config: Optional[str] = None, split: Optional[str] = "train",
                 target_feature: Optional[str] = None,
                 loading_options: Optional[Dict] = None):
        """Create a Dataset.

        Args:
            dataset: Name or path of the dataset. Use the path for local datasets.
            load_from: None for given datasets, one of "huggingface" or "local" otherwise.
            config: Configuration, used for Huggingface datasets.
            split: Split, used for Huggingface datasets.
            target_feature: Target feature of the dataset.
            loading_options: Keyword arguments provided to pandas.read_X for loading the dataset if local.
        """
        warnings.simplefilter(action="ignore", category=UserWarning)

        if load_from is not None:
            match load_from:
                case "huggingface":
                    df = load_huggingface_dataset(dataset, config)[split].to_pandas().infer_objects()
                    categorical_features = [f for f in df.columns
                                            if df.dtypes[f].name in ("object", "string", "category")]
                    df = df.astype({f: "category" for f in categorical_features})
                    super(Dataset, self).__init__(data=df)

                case "local":
                    suffix = dataset.split("."[-1])
                    loading_options = loading_options if loading_options is not None else dict()
                    match suffix:
                        case "csv":
                            df = pandas.read_csv(dataset, **loading_options).infer_objects()
                        case "json":
                            df = pandas.read_json(dataset, **loading_options).infer_objects()
                        case _:
                            raise ValueError(f"Unknown dataset extension: {suffix}")

                    categorical_features = [f for f in df.columns
                                            if df.dtypes[f].name in ("object", "string", "category")]
                    df = df.astype({f: "category" for f in categorical_features})
                    super(Dataset, self).__init__(data=df)
                case _:
                    raise NotImplementedError()
        else:
            if isinstance(dataset, Dataset):
                df = copy.deepcopy(dataset)
                super(Dataset, self).__init__(data=df.to_pandas())
                df.copy_metadata_from(dataset)

            elif isinstance(dataset, DataFrame):
                categorical_features = [f for f in dataset.columns
                                        if dataset.dtypes[f].name in ("object", "string", "category")]
                df = dataset.astype({f: "category" for f in categorical_features})
                super(Dataset, self).__init__(data=df)

            elif isinstance(dataset, numpy.ndarray):
                # can't use PA array since it can't handle non-numeric numpy arrays
                df = DataFrame(dataset).infer_objects()
                categorical_features = [f for f in df.dtypes
                                        if df.dtypes[f].name in ("object", "string", "category")]
                df = dataset.astype({f: "category" for f in categorical_features})
                super(Dataset, self).__init__(data=df)

            elif isinstance(dataset, ArrowDataset):
                df = dataset.to_pandas().infer_objects()
                categorical_features = [f for f in df.dtypes
                                        if df.dtypes[f].name in ("object", "string", "category")]
                df = df.astype({f: "category" for f in categorical_features})
                super(Dataset, self).__init__(data=df)
            else:
                raise ValueError(f"Invalid type: {type(dataset)}")

        self.target_feature = target_feature
        self.one_hot_encoding_dictionary = dict()
        self.bins_encoding_dictionary = dict()
        self.target_encoding_dictionary = dict()

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        return len(self) == len(other) \
            and all(self.iloc[i] == other.iloc[i] for i in range(len(self)))

    def __hash__(self):
        return hash(str(self.to_pandas()))

    def __copy__(self, **kwargs) -> Dataset:
        dataset = Dataset(self.copy())
        dataset.copy_metadata_from(self)

        return dataset

    def __deepcopy__(self, memodict=None) -> Dataset:
        dataset = Dataset(self.copy())
        dataset.copy_metadata_from(self)

        return dataset

    def __delitem__(self, key):
        new_dataframe = copy.deepcopy(self)
        new_dataframe.drop(key, axis="index", inplace=True)
        new_dataframe = Dataset(new_dataframe)
        new_dataframe.copy_metadata_from(self)

        return new_dataframe

    ################
    ## Conversion ##
    ################
    def to_pandas(self):
        return DataFrame(self)

    def to_array(self):
        return self.values

    def to_list(self):
        return [tuple(row) for row in self.itertuples()]

    ###########
    ## Types ##
    ###########

    def copy_metadata_from(self, dataset: Dataset) -> Dataset:
        """Copy metadata from the given dataset.

        Args:
            dataset: The dataset to copy metadata from.

        Returns:
            This dataset, with overwritten metadata.
        """
        self.bins_encoding_dictionary = copy.deepcopy(dataset.bins_encoding_dictionary)
        self.one_hot_encoding_dictionary = copy.deepcopy(dataset.one_hot_encoding_dictionary)
        self.target_encoding_dictionary = copy.deepcopy(dataset.target_encoding_dictionary)
        self.target_feature = copy.deepcopy(dataset.target_feature)

        return self

    def astype(self, dtype, copy: bool = True, errors: IgnoreRaise = "raise") -> Dataset:
        processed_dataset = Dataset(super(Dataset, self).astype(dtype))
        processed_dataset.copy_metadata_from(self)

        return processed_dataset

    #####################
    ## Pandas override ##
    ####################
    def update(self, other, join: str = "left", overwrite: bool = True, filter_func=None, errors: str = "ignore") -> Dataset:
        d = Dataset(self.to_pandas().update(other, join, overwrite, filter_func, errors))
        d.copy_metadata_from(self)

        return d

    def dot(self, other: DataFrame | Index | ArrayLike) -> Dataset:
        d = Dataset(self.to_pandas().dot(other))
        d.copy_metadata_from(self)

        return d

    @classmethod
    def from_dict(cls, data: dict, orient: str = "columns", dtype: Dtype | None = None,
                  columns: Axes | None = None) -> Dataset:
        return Dataset(DataFrame.from_dict(data, orient, dtype, columns))

    @classmethod
    def from_records(cls, data, index=None, exclude=None, columns=None, coerce_float: bool = False,
                     nrows: int | None = None) -> Dataset:
        return Dataset(DataFrame.from_records(data, index, exclude, columns, coerce_float, nrows))

    def transpose(self, *args, copy: bool = False) -> Dataset:
        return Dataset(super(Dataset, self).transpose(*args, copy=copy)).copy_metadata_from(self)

    def query(self, expr: str, *, inplace: Literal[False] = ..., **kwargs) -> Dataset:
        return Dataset(super(Dataset, self).query(self, expr, inplace, **kwargs)).copy_metadata_from(self)

    def select_dtypes(self, include=None, exclude=None) -> Dataset:
        return Dataset(super(Dataset, self).select_dtypes(include, exclude)).copy_metadata_from(self)

    def assign(self, **kwargs) -> Dataset:
        return Dataset(super(Dataset, self).assign(**kwargs)).copy_metadata_from(self)

    def align(
            self,
            other: DataFrame,
            join: Literal["outer", "inner", "left", "right"] = "outer",
            axis: Axis | None = None,
            level: Level = None,
            copy: bool = True,
            fill_value=None,
            method: FillnaOptions | None = None,
            limit: int | None = None,
            fill_axis: Axis = 0,
            broadcast_axis: Axis | None = None,
    ) -> Dataset:
        return Dataset(super(Dataset, self).align(other, join, axis, level, copy, fill_value, method, limit, fill_axis,
                                  broadcast_axis)).copy_metadata_from(self)

    def set_axis(
            self,
            labels,
            axis: Axis,
            copy: bool,
    ) -> Dataset:
        return Dataset(super(Dataset, self).set_axis(labels, axis, copy)).copy_metadata_from(self)

    def reindex(self, *args, **kwargs) -> Dataset:
        return Dataset(super(Dataset, self).reindex(*args, **kwargs)).copy_metadata_from(self)

    def drop(
            self,
            labels,
            axis,
            index,
            columns,
            level,
            inplace,
            errors: IgnoreRaise = ...,
    ) -> None:
        super(Dataset, self).drop(labels, axis, index, columns, level, inplace, errors)
        if axis in ("columns", 1):
            if labels == self.target_feature:
                self.target_feature = None
            if isinstance(labels, str):
                dropped_labels = [labels]
            else:
                dropped_labels = labels
            for l in dropped_labels:
                if l in self.one_hot_encoding_dictionary:
                    del self.one_hot_encoding_dictionary[l]
                if l in self.target_encoding_dictionary:
                    del self.target_encoding_dictionary[l]
                if l in self.bins_encoding_dictionary:
                    del self.bins_encoding_dictionary[l]

    def rename(
            self,
            mapper,
            index,
            columns,
            axis,
            copy,
            inplace,
            level,
            errors,
    ) -> None:
        super(Dataset, self).rename(mapper, index, columns, axis, copy, inplace, level, errors)

    def fillna(
            self,
            value,
            method,
            axis,
            inplace,
            limit,
            downcast,
    ) -> Dataset:
        return Dataset(super(Dataset, self).fillna(value, method, axis, inplace, limit, downcast)).copy_metadata_from(self)

    def replace(
            self,
            to_replace,
            value,
            inplace,
            limit,
            regex,
            method,
    ) -> Dataset:
        return Dataset(super(Dataset, self).replace(to_replace, value, inplace, limit, regex, method)).copy_metadata_from(self)

    def shift(
            self,
            periods = 1,
            freq = None,
            axis =  0,
            fill_value = lib.no_default,
    ) -> Dataset:
        return Dataset(super(Dataset, self).shift(periods, freq, axis, fill_value)).copy_metadata_from(self)

    def set_index(
            self,
            keys,
            drop,
            append,
            inplace,
            verify_integrity,
    ) -> Dataset:
        return Dataset(super(Dataset, self).set_index(keys, drop, append, inplace, verify_integrity)).copy_metadata_from(self)

    def reset_index(
            self,
            level,
            drop,
            inplace,
            col_level,
            col_fill,
            allow_duplicates,
            names = None,
    ) -> Dataset:
        return Dataset(super(Dataset, self).reset_index(level, drop, inplace, col_level, col_fill,
                                        allow_duplicates, names)).copy_metadata_from(self)

    def dropna(
            self,
            axis,
            how,
            thresh,
            subset,
            inplace,
    ) -> Dataset:
        return Dataset(super(Dataset, self).dropna(axis, how, thresh, subset, inplace)).copy_metadata_from(self)

    def drop_duplicates(
            self,
            subset = None,
            keep: Literal["first", "last", False] = "first",
            inplace = False,
            ignore_index = False,
    ) -> Dataset | None:
        return Dataset(super(Dataset, self).drop_duplicates(subset, keep, inplace, ignore_index)).copy_metadata_from(self)

    def sort_values(
            self,
            by,
            axis,
            ascending,
            inplace,
            kind,
            na_position,
            ignore_index,
            key,
    ) -> Dataset:
        return Dataset(super(Dataset, self).sort_values(by, axis, ascending, inplace, kind, na_position, ignore_index,
                                           key)).copy_metadata_from(self)

    def sort_index(
            self,
            *,
            axis,
            level,
            ascending,
            inplace,
            kind,
            na_position,
            sort_remaining,
            ignore_index,
            key,
    ) -> Dataset:
        return Dataset(super(Dataset, self).sort_index(axis, level, ascending, inplace, kind, na_position, sort_remaining,
                                          ignore_index, key)).copy_metadata_from(self)

    def nlargest(self, n: int, columns, keep: str = "first") -> Dataset:
        return Dataset(super(Dataset, self).n_largest(n, columns, keep)).copy_metadata_from(self)

    def nsmallest(self, n: int, columns, keep: str = "first") -> Dataset:
        return Dataset(super(Dataset, self).n_smallest(n, columns, keep)).copy_metadata_from(self)

    def swaplevel(self, i: Axis = -2, j: Axis = -1, axis: Axis = 0) -> Dataset:
        return Dataset(super(Dataset, self).swaplevel(i, j, axis)).copy_metadata_from(self)

    def reorder_levels(self, order: Sequence[Axis], axis: Axis = 0) -> Dataset:
        return Dataset(super(Dataset, self).reorder_levels(order, axis)).copy_metadata_from(self)

    def __divmod__(self, other) -> tuple[DataFrame, DataFrame]:
        res = super(Dataset, self).__divmod__(other)
        return Dataset(res[0]).copy_metadata_from(self), Dataset(res[1]).copy_metadata_from(self)

    def __rdivmod__(self, other) -> tuple[DataFrame, DataFrame]:
        res = super(Dataset, self).__rdivmod__(other)
        return Dataset(res[0]).copy_metadata_from(self), Dataset(res[1]).copy_metadata_from(self)

    def compare(
            self,
            other: DataFrame,
            align_axis: Axis = 1,
            keep_shape: bool = False,
            keep_equal: bool = False,
            result_names = ("self", "other"),
    ):
        return Dataset(super(Dataset, self).compare(other, align_axis, keep_shape, keep_equal, result_names)).copy_metadata_from(self)

    def combine(
        self,
        other: DataFrame,
        func: Callable[[Series, Series], Series | Hashable],
        fill_value=None,
        overwrite: bool = True,
    ) -> Dataset:
        return Dataset(super(Dataset, self).combine(other, func, fill_value, False)).copy_metadata_from(self)

    def combine_first(self, other: DataFrame) -> Dataset:
        return Dataset(super(Dataset, self).combine_first(other)).copy_metadata_from(self)

    def pivot(self, index=None, columns=None, values=None) -> Dataset:
        return Dataset(super(Dataset, self).pivot(index, columns, values)).copy_metadata_from(self)

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
        observed=False,
        sort=True,
    ) -> Dataset:
        return Dataset(super(Dataset, self).pivot_table(values, index, columns, aggfunc, fill_value, margins, dropna, margins_name,
                                        observed, sort)).copy_metadata_from(self)

    def explode(
        self,
        column,
        ignore_index: bool = False,
    ) -> Dataset:
        return Dataset(super(Dataset, self).explode(column, ignore_index)).copy_metadata_from(self)

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level: Level = None,
        ignore_index: bool = True,
    ) -> Dataset:
        return Dataset(super(Dataset, self).melt(id_vars, value_vars, var_name, value_name, col_level,
                                 ignore_index)).copy_metadata_from(self)

    def diff(self, periods: int = 1, axis: Axis = 0) -> Dataset:
        return Dataset(super(Dataset, self).diff(periods, axis)).copy_metadata_from(self)

    def transform(
        self, func, axis: Axis = 0, *args, **kwargs
    ) -> Dataset:
        return Dataset(super(Dataset, self).transform(func, axis, *args, **kwargs)).copy_metadata_from(self)

    def applymap(
        self, func, na_action: str | None = None, **kwargs
    ) -> Dataset:
        return Dataset(super(Dataset, self).applymap(func, na_action, **kwargs)).copy_metadata_from(self)

    def append(
        self,
        other,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ) -> Dataset:
        return Dataset(super(Dataset, self).append(other, ignore_index, verify_integrity, sort)).copy_metadata_from(self)

    def join(
        self,
        other: DataFrame | Series | list[DataFrame | Series],
        on: IndexLabel | None = None,
        how: str = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
        validate: str | None = None,
    ) -> Dataset:
        return Dataset(super(Dataset, self).join(other, on, how, lsuffix, rsuffix, sort, validate)).copy_metadata_from(self)

    def merge(
        self,
        right: DataFrame | Series,
        how: str = "inner",
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: Suffixes = ("_x", "_y"),
        copy: bool = True,
        indicator: bool = False,
        validate: str | None = None,
    ) -> Dataset:
        return Dataset(super(Dataset, self).merge(right, how, on, left_on, left_index, right_index, sort, suffixes,
                                  copy, indicator, validate)).copy_metadata_from(self)

    def round(
        self, decimals: int | dict[IndexLabel, int] | Series = 0, *args, **kwargs
    ) -> Dataset:
        return Dataset(super(Dataset, self).round(decimals, *args, **kwargs)).copy_metadata_from(self)

    def corr(
        self,
        method: str | Callable[[numpy.ndarray, numpy.ndarray], float] = "pearson",
        min_periods: int = 1,
        numeric_only: bool | lib.NoDefault = lib.no_default,
    ) -> Dataset:
        return Dataset(super(Dataset, self).corr(method, min_periods, numeric_only)).copy_metadata_from(self)

    def cov(
        self,
        min_periods: int | None = None,
        ddof: int | None = 1,
        numeric_only: bool | lib.NoDefault = lib.no_default,
    ) -> Dataset:
        return Dataset(super(Dataset, self).cov(min_periods, ddof, numeric_only)).copy_metadata_from(self)

    def mode(
        self, axis: Axis = 0, numeric_only: bool = False, dropna: bool = True
    ) -> Dataset:
        return Dataset(super(Dataset, self).mode(axis, numeric_only, dropna)).copy_metadata_from(self)

    def quantile(
            self,
            q: AnyArrayLike | Sequence[float],
            axis: Axis,
            numeric_only: bool | lib.NoDefault,
            interpolation: QuantileInterpolation,
    ) -> Series | DataFrame:
        return Dataset(super(Dataset, self).quantile(q, axis, numeric_only, interpolation)).copy_metadata_from(self)

    def asfreq(
        self,
        freq: Frequency,
        method: FillnaOptions | None = None,
        how: str | None = None,
        normalize: bool = False,
        fill_value: Hashable = None,
    ) -> Dataset:
        return Dataset(super(Dataset, self).asfreq(freq, method, how, normalize, fill_value)).copy_metadata_from(self)

    def to_timestamp(
        self,
        freq: Frequency | None = None,
        how: str = "start",
        axis: Axis = 0,
        copy: bool = True,
    ) -> Dataset:
        return Dataset(super(Dataset, self).to_timestamp(freq, how, axis, copy)).copy_metadata_from(self)

    def to_period(
        self, freq: Frequency | None = None, axis: Axis = 0, copy: bool = True
    ) -> Dataset:
        return Dataset(self.to_period(freq, axis, copy)).copy_metadata_from(self)

    def isin(self, values: Series | DataFrame | Sequence | Mapping) -> Dataset:
        return Dataset(super(Dataset, self).isin(values)).copy_metadata_from(self)

    def ffill(
        self,
        axis,
        inplace: Literal[False],
        limit: None | int,
        downcast: dict | None,
    ) -> Dataset:
        return Dataset(super(Dataset, self).ffill(axis, inplace, limit, downcast)).copy_metadata_from(self)

    def bfill(
        self,
        *,
        axis: None | Axis,
        inplace: Literal[False],
        limit: None | int,
        downcast,
    ) -> Dataset:
        return Dataset(super(Dataset, self).bfill(axis, inplace, limit, downcast)).copy_metadata_from(self)

    def clip(
        self: DataFrame,
        lower: float | None = None,
        upper: float | None = None,
        axis: Axis | None = None,
        inplace: bool = False,
        *args,
        **kwargs,
    ) -> Dataset | None:
        return Dataset(super(DataFrame, self).clip(lower, upper, axis, False, *args, **kwargs)).copy_metadata_from(self)

    def interpolate(
        self: DataFrame,
        method: str = "linear",
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: str | None = None,
        limit_area: str | None = None,
        downcast: str | None = None,
        **kwargs,
    ) -> Dataset | None:
        return Dataset(super(DataFrame, self).interpolate(method, axis, limit, False, limit_direction, limit_area,
                                        downcast, **kwargs)).copy_metadata_from(self)

    def where(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> Dataset:
        return Dataset(super(Dataset, self).where(cond, other, inplace, axis, level, errors, try_cast)).copy_metadata_from(self)

    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> Dataset:
        return Dataset(super(Dataset, self).mask(cond, other, inplace, axis, level, errors, try_cast)).copy_metadata_from(self)

    ################
    ## Data stuff ##
    ################
    def train_test_split(self, test_size: float = 0.2,
                         stratify: Optional[str | Sequence[str]] = None) -> Tuple[numpy.array, numpy.array]:
        """Split this dataset into two, possibly stratifying with a set of given features.

        Args:
            test_size: Size of the test set, in a [0, 1] percentage.
            stratify: Optional feature(s) to stratify the split.

        Returns:
            Indexes of each split.
        """
        df = self.to_pandas()
        nr_records = df.shape[0]
        if stratify is not None:
            unique_stratify_values = df[stratify].unique()
            stratify_balance = numpy.unique(df[stratify].values, return_counts=True)
            stratify_balance = stratify_balance / len(self)
            indexes_per_values = [numpy.argwhere(df[stratify].values == stratify_value).squeeze()
                                  for stratify_value in unique_stratify_values]
            indexes = tuple([numpy.random.choice(value_indexes, int(balance * nr_records))
                             for value_indexes, balance in zip(indexes_per_values, stratify_balance)])
        else:
            train_indexes = numpy.random.choice(numpy.arange(nr_records), nr_records * (1 - test_size))
            test_indexes = numpy.array([i for i in range(nr_records) if i not in train_indexes])
            indexes = (train_indexes, test_indexes)

        return indexes
