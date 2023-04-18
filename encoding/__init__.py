from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Sequence, TypeVar

import numpy

T = TypeVar("T")
S = TypeVar("S")


class Binner(ABC):
    def __init__(self, encoding_map: Optional[Dict] = None):
        self.encoding_map = encoding_map
        self.is_fit = False

    @abstractmethod
    def fit(self, data: T, **kwargs) -> Binner:
        """Fit this Binner on `data`.

        Args:
            data: The data to fit this binner on

        Returns:
            This fit Binner.
        """
        pass

    @abstractmethod
    def bin(self, data: T, **kwargs) -> Sequence[S]:
        """Bin the given data.

        Args:
            data: The data to bin.

        Returns:
              The binned data.
        """
        pass

    def fit_bin(self, fit_data: T, bin_data: T, **kwargs) -> Sequence[S]:
        """Fit this Binner on `fit_data`, then bin `bin_data`.

        Args:
            fit_data: The data to fit this binner.
            bin_data: The data to bin.
        Returns:
            The binned data.
        """
        self.fit(fit_data, **kwargs)
        self.is_fit = True
        binned_data = self.bin(bin_data, **kwargs)

        return binned_data

    def one_hot_bin(self, data: T, **kwargs) -> numpy.array:
        """Map the given data into bins, then encode them in a one-hot fashion.

        Args:
            data: The values to map.
            kwargs: Keyword arguments.

        Returns:
            The binned `data`, bins encoded in a one-hot fashion.
        """
        base = numpy.zeros(len(data), len(self.encoding_map)).astype(int)
        binned_values = self.bin(data, **kwargs)
        for i, bin_value in enumerate(binned_values):
            base[i, bin_value] = 1

        return base

    def one_hot_fit_bin(self, fit_data: T, bin_data: T, **kwargs) -> Sequence[S]:
        """Fit this Binner on `fit_data`, then bin `bin_data`.

        Args:
            fit_data: The data to fit this binner.
            bin_data: The data to bin.
        Returns:
           The binned data.
        """
        self.fit(fit_data, **kwargs)

        base = numpy.zeros((len(bin_data), len(self.encoding_map))).astype(int)
        binned_values = self.bin(bin_data, **kwargs)
        for i, bin_value in enumerate(binned_values):
            base[i, bin_value] = 1

        return base

    def transform(self, data: T, **kwargs) -> Sequence[S]:
        """sklearn-compatible interface.
        Bin the given data.

        Args:
            data: The data to bin.

        Returns:
              The binned data.
        """
        return self.bin(data, **kwargs)

    def fit_transform(self, fit_data: T, bin_data: T, **kwargs) -> Sequence[S]:
        """sklearn interface.
        Fit this Binner on `fit_data`, then bin `bin_data`.

        Args:
            fit_data: The data to fit this binner.
            bin_data: The data to bin.
        Returns:
            The binned data.
        """
        return self.fit_bin(fit_data, bin_data, **kwargs)

    def one_hot_transform(self, data: T, **kwargs) -> numpy.array:
        """sklearn interface.
        Map the given data into bins, then encode them in a one-hot fashion.

        Args:
            data: The values to map.
            kwargs: Keyword arguments.

        Returns:
            The binned `data`, bins encoded in a one-hot fashion.
        """
        return self.one_hot_bin(data, **kwargs)


class NumericBinner(Binner, ABC):
    """A Binner, only limited to numerical data."""
    def __init__(self):
        super(NumericBinner, self).__init__()
