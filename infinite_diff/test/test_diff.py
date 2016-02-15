#! /usr/bin/env python
"""Tests of finite differencing module."""
from __future__ import division

import sys
import unittest

from infinite_diff import FiniteDiff, FwdDiff, BwdDiff, CenDiff
import numpy as np
import xarray as xr

from . import InfiniteDiffTestCase


class TestFiniteDiff(InfiniteDiffTestCase):
    def setUp(self):
        super(TestFiniteDiff, self).setUp()
        self.cls = FiniteDiff
        self.fd_ones_trunc = [self.cls(self.ones_trunc[n], self.dim)
                              for n in range(self.array_len)]

    def test_check_spacing(self):
        self.assertRaises(ValueError, self.cls._check_spacing,
                          **dict(spacing=0))
        self.assertRaises(TypeError, self.cls._check_spacing,
                          **dict(spacing=1.1))
        # Test that no exception is raised for valid input.
        [self.cls._check_spacing(spacing=n) for n in range(1, 20)]

    def test_check_array_len(self):
        for n, fd in enumerate(self.fd_ones_trunc):
            self.assertRaises(ValueError, fd._check_arr_len,
                              **dict(spacing=len(fd.arr[self.dim])))
        fd_len0 = self.cls(self.ones.isel(**{self.dim: 0}), self.dim)
        self.assertRaises(ValueError, fd_len0._check_arr_len)

    def test_slice_arr_dim(self):
        slice_ = slice(1, -2)
        actual = self.cls(self.ones, self.dim)._slice_arr_dim(slice_, None)
        self.assertDatasetIdentical(actual,
                                    self.ones.isel(**{self.dim: slice_}))

    def test_reverse_dim(self):
        values = np.arange(self.array_len)
        arr = xr.DataArray(values, dims=[self.dim],
                           coords={self.dim: values})
        actual = self.cls(arr, self.dim)._reverse_dim()
        desired = xr.DataArray(values[::-1], dims=[self.dim],
                               coords={self.dim: values[::-1]})
        self.assertDatasetIdentical(actual, desired)

    def test_diff_not_implemented(self):
        self.assertRaises(NotImplementedError,
                          self.cls(self.random, self.dim).diff)


class FwdDiffTestCase(InfiniteDiffTestCase):
    def setUp(self):
        super(FwdDiffTestCase, self).setUp()
        self.cls = FwdDiff
        self.is_bwd = False


class TestFwdDiff(FwdDiffTestCase):
    def test_diff_output_coords(self):
        for n in range(self.array_len - 1):
            actual = self.cls(self.ones, self.dim).diff(spacing=n+1).coords
            trunc = slice(n+1, None) if self.is_bwd else slice(0, -(n+1))
            desired = self.ones.isel(**{self.dim: trunc}).coords
            assert actual.to_dataset().identical(desired.to_dataset())

    def test_diff_zero_slope_varied_arr_len(self):
        for n, ones in enumerate(self.ones_trunc[:-2]):
            actual = self.cls(ones, self.dim).diff()
            desired = self.zeros_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_zero_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self.cls(self.ones, self.dim).diff(spacing=n+1)
            desired = self.zeros_trunc[n]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_arr_len(self):
        for n, arange in enumerate(self.arange_trunc[:-2]):
            actual = self.cls(arange, self.dim).diff()
            desired = self.ones_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self.cls(self.arange, self.dim).diff(spacing=n+1)
            desired = (n+1)*ones
            self.assertDatasetIdentical(actual, desired)

    def _compar_to_diff(self, arr):
        label = 'upper' if self.is_bwd else 'lower'
        actual = self.cls(arr, self.dim).diff()
        desired = arr.diff(self.dim, n=1, label=label)
        self.assertDatasetIdentical(actual, desired)

    def test_diff_misc_slopes(self):
        for arr in [self.ones, self.zeros, self.arange, self.random]:
            self._compar_to_diff(arr)


class TestBwdDiff(TestFwdDiff):
    def setUp(self):
        super(TestBwdDiff, self).setUp()
        self.cls = BwdDiff
        self.is_bwd = True

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, None)})
                            for n in range(self.array_len)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, None)})
                           for n in range(self.array_len)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]


class CenDiffTestCase(InfiniteDiffTestCase):
    def setUp(self):
        super(CenDiffTestCase, self).setUp()
        self.cls = CenDiff

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, -(n+1))})
                            for n in range(self.array_len // 2 - 1)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, -(n+1))})
                           for n in range(self.array_len // 2 - 1)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]


class TestCenDiff(CenDiffTestCase):
    def test_diff_output_coords(self):
        for n in range(self.array_len // 2 - 1):
            actual = self.cls(self.ones, self.dim).diff(spacing=n+1).coords
            trunc = slice(n+1, -(n+1))
            desired = self.ones.isel(**{self.dim: trunc}).coords
            assert actual.to_dataset().identical(desired.to_dataset())

    def test_diff_zero_slope_varied_arr_len(self):
        for n, ones in enumerate(self.ones_trunc[:-2]):
            actual = self.cls(ones, self.dim).diff()
            desired = self.zeros_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_zero_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self.cls(self.ones, self.dim).diff(spacing=n+1)
            desired = self.zeros_trunc[n]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_arr_len(self):
        for n, arange in enumerate(self.arange_trunc[:-2]):
            actual = self.cls(arange, self.dim).diff()
            desired = 2*self.ones_trunc[n+1]
            self.assertDatasetIdentical(actual, desired)

    def test_diff_const_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self.cls(self.arange, self.dim).diff(spacing=n+1)
            desired = 2*(n+1)*ones
            self.assertDatasetIdentical(actual, desired)

    def _compar_to_diff(self, arr):
        actual = self.cls(arr, self.dim).diff()
        desired_values = (arr.isel(**{self.dim: slice(2, None)}).values -
                          arr.isel(**{self.dim: slice(None, -2)})).values
        desired = xr.DataArray(desired_values, dims=actual.dims,
                               coords=actual.coords)
        self.assertDatasetIdentical(actual, desired)

    def test_diff_misc_slopes(self):
        for arr in [self.ones, self.zeros, self.arange, self.random]:
            self._compar_to_diff(arr)


if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: more of private utility methods
# TODO: OneSidedDiff class
# TODO: non-default coord values (shouldn't affect diffs on arrays)
