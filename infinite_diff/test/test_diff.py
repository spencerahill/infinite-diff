#! /usr/bin/env python
"""Tests of finite differencing module."""
from __future__ import division

import sys
import unittest

from infinite_diff import FiniteDiff, FwdDiff, BwdDiff, CenDiff
import numpy as np
import xarray as xr

from . import InfiniteDiffTestCase


class DiffSharedTests(object):
    def test_slice_arr_dim(self):
        slice_ = slice(1, -2)
        arr = self.ones
        actual = self._DIFF_CLS(arr, self.dim)._slice_arr_dim(slice_, arr)
        self.assertDatasetIdentical(actual, arr[{self.dim: slice_}])

    def test_reverse_dim(self):
        values = np.arange(self.array_len)
        arr = xr.DataArray(values, dims=[self.dim], coords={self.dim: values})
        actual = self._DIFF_CLS(arr, self.dim)._reverse_dim(arr)
        desired = xr.DataArray(values[::-1], dims=[self.dim],
                               coords={self.dim: values[::-1]})
        self.assertDatasetIdentical(actual, desired)

    def test_diff(self):
        self.assertNotImplemented(self.diff_obj.diff)
        self.assertNotImplemented(self.diff_obj._diff)
        self.assertNotImplemented(self.diff_obj.diff_rev)
        self.assertNotImplemented(self.diff_obj._diff_rev)


class FiniteDiffTestCase(InfiniteDiffTestCase):
    _DIFF_CLS = FiniteDiff

    def setUp(self):
        super(FiniteDiffTestCase, self).setUp()
        self.spacing = 1
        self.arr = self.ones
        self.diff_obj = self._DIFF_CLS(self.arr, self.dim,
                                       spacing=self.spacing)
        self.fd_ones_trunc = [self._DIFF_CLS(self.ones_trunc[n], self.dim)
                              for n in range(self.array_len)]


class TestFiniteDiff(DiffSharedTests, FiniteDiffTestCase):
    pass


class FwdDiffTestCase(FiniteDiffTestCase):
    _DIFF_CLS = FwdDiff
    _IS_BWD = False

    def setUp(self):
        super(FwdDiffTestCase, self).setUp()


class TestFwdDiff(FwdDiffTestCase):
    def test_diff_output_coords(self):
        for n in range(self.array_len - 1):
            actual = self._DIFF_CLS(self.ones, self.dim,
                                    spacing=n+1).diff()
            trunc = slice(n+1, None) if self._IS_BWD else slice(0, -(n+1))
            desired = self.ones[{self.dim: trunc}]
            self.assertCoordsIdentical(actual, desired)

    def test_diff_zero_slope_varied_arr_len(self):
        for n, ones in enumerate(self.ones_trunc[:-2]):
            actual = self._DIFF_CLS(ones, self.dim).diff()
            desired = self.zeros_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_zero_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self._DIFF_CLS(self.ones, self.dim, spacing=n+1).diff()
            desired = self.zeros_trunc[n]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_arr_len(self):
        for n, arange in enumerate(self.arange_trunc[:-2]):
            actual = self._DIFF_CLS(arange, self.dim).diff()
            desired = self.ones_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self._DIFF_CLS(self.arange, self.dim, spacing=n+1).diff()
            desired = (n+1)*ones
            self.assertDatasetIdentical(actual, desired)

    def _compar_to_diff(self, arr):
        label = 'upper' if self._IS_BWD else 'lower'
        actual = self._DIFF_CLS(arr, self.dim).diff()
        desired = arr.diff(self.dim, n=1, label=label)
        self.assertDatasetIdentical(actual, desired)

    def test_diff_misc_slopes(self):
        for arr in [self.ones, self.zeros, self.arange, self.random]:
            self._compar_to_diff(arr)


class BwdDiffTestCase(FwdDiffTestCase):
    _DIFF_CLS = BwdDiff
    _IS_BWD = True

    def setUp(self):
        super(BwdDiffTestCase, self).setUp()
        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, None)})
                            for n in range(self.array_len)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, None)})
                           for n in range(self.array_len)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]


class TestBwdDiff(TestFwdDiff, BwdDiffTestCase):
    pass


class CenDiffTestCase(FiniteDiffTestCase):
    _DIFF_CLS = CenDiff

    def setUp(self):
        super(CenDiffTestCase, self).setUp()
        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, -(n+1))})
                            for n in range(self.array_len // 2 - 1)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, -(n+1))})
                           for n in range(self.array_len // 2 - 1)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]


class TestCenDiff(DiffSharedTests, CenDiffTestCase):
    def test_diff_output_coords(self):
        for n in range(self.array_len // 2 - 1):
            actual = self._DIFF_CLS(self.ones, self.dim,
                                    spacing=n+1).diff()
            trunc = slice(n+1, -(n+1))
            desired = self.ones[{self.dim: trunc}]
            self.assertCoordsIdentical(actual, desired)

    def test_diff_zero_slope_varied_arr_len(self):
        for n, ones in enumerate(self.ones_trunc[:-2]):
            actual = self._DIFF_CLS(ones, self.dim).diff()
            desired = self.zeros_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_zero_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self._DIFF_CLS(self.ones, self.dim, spacing=n+1).diff()
            desired = self.zeros_trunc[n]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_arr_len(self):
        for n, arange in enumerate(self.arange_trunc[:-2]):
            actual = self._DIFF_CLS(arange, self.dim).diff()
            desired = 2*self.ones_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self._DIFF_CLS(self.arange, self.dim, spacing=n+1).diff()
            desired = 2*(n+1)*ones
            self.assertDatasetIdentical(actual, desired)

    def test_diff_fill_edge(self):
        fills = [False, 'left', 'right', 'both', True]
        truncs = [slice(1, -1), slice(0, -1), slice(1, None),
                  slice(None, None), slice(None, None)]
        for fill, trunc in zip(fills, truncs):
            actual = self._DIFF_CLS(self.arange, self.dim, spacing=1,
                                    fill_edge=fill).diff()
            desired = self.arange[{self.dim: trunc}]
            self.assertCoordsIdentical(actual, desired)

    def _compar_to_diff(self, arr):
        actual = self._DIFF_CLS(arr, self.dim).diff()
        desired_values = (arr.isel(**{self.dim: slice(2, None)}).values -
                          arr.isel(**{self.dim: slice(None, -2)})).values
        desired = xr.DataArray(desired_values, dims=actual.dims,
                               coords=actual.coords)
        self.assertDatasetIdentical(actual, desired)

    def test_diff(self):
        for arr in [self.ones, self.zeros, self.arange, self.random]:
            self._compar_to_diff(arr)


if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: more of private utility methods
# TODO: OneSidedDiff class
# TODO: non-default coord values (shouldn't affect diffs on arrays)
