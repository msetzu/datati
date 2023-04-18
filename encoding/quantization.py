"""Quantization measures to map sets of values to bins."""
from __future__ import annotations

import numpy

from encoding import NumericBinner


class Quantiler(NumericBinner):
    """Quantization of numerical values into bins given by quantiles."""
    def __init__(self, n_quantiles: int = 4):
        super().__init__()
        self.n_quantiles = n_quantiles

    def fit(self, values: numpy.ndarray, **kwargs) -> Quantiler:
        """Fit this Quantizer to the given `values`.

        Args:
            values: The values to fit.

        Returns:
           This fit Quantizer.
        """
        if not(values.dtype.name.startswith("int") or values.dtype.name.startswith("float")):
            self.is_fit = True
            self.encoding_map = dict()

            return self

        desired_quantiles = numpy.arange(0., 1. + 1 / self.n_quantiles, step=1 / self.n_quantiles)
        quantiles = numpy.quantile(values, q=desired_quantiles)

        self.qs = quantiles
        self.encoding_map = [(self.qs[i], self.qs[i + 1]) for i in range(1, len(self.qs) - 1)]
        self.encoding_map = [(-numpy.inf, self.qs[0])] + self.encoding_map + [(self.qs[-1], +numpy.inf)]

        self.is_fit = True

        return self

    def bin(self, values: numpy.ndarray, **kwargs) -> numpy.ndarray:
        """Map the given `values` in the computed quantiles.

        Args:
            values: The values to map.

        Returns:
            The transformed values as per the computed quantiles.
        """
        return numpy.array([self.value_to_quantile(v, self.qs) for v in values])

    def value_to_quantile(self, value: float, qs: numpy.ndarray) -> int:
        upper_bounds = numpy.argwhere(value <= qs)
        if upper_bounds.size == 0:
            # clip up
            q = qs.size - 1
        else:
            q = max(numpy.argwhere(value <= qs)[0].item() - 1, 0)

        return q
