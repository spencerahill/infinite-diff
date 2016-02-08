#! /usr/bin/env python
"""Tests"""
import sys
import unittest

from infinite_diff import FiniteDiff, FwdDiff, BwdDiff, CenDiff
import numpy as np
import xarray as xr

from . import InfiniteDiffTestCase


class TestFiniteDiff(InfiniteDiffTestCase):
    def setUp(self):
        super(TestFiniteDiff, self).setUp()
        self.fd = FiniteDiff(self.ones, self.dim)
        self.fd_ones_trunc = [FiniteDiff(self.ones_trunc[n], self.dim)
                              for n in range(self.array_len)]

    def test_check_spacing(self):
        self.assertRaises(ValueError, FiniteDiff._check_spacing,
                          **dict(spacing=0))
        self.assertRaises(TypeError, FiniteDiff._check_spacing,
                          **dict(spacing=1.1))
        # Test that no exception is raised for valid input.
        [FiniteDiff._check_spacing(spacing=n) for n in range(1, 20)]

    def test_check_array_len(self):
        for n, fd in enumerate(self.fd_ones_trunc):
            self.assertRaises(ValueError, fd._check_arr_len,
                              **dict(spacing=len(fd.arr[self.dim])))

        fd_len0 = FiniteDiff(self.ones.isel(**{self.dim: 0}), self.dim)
        self.assertRaises(ValueError, fd_len0._check_arr_len)

    @unittest.skip("Needs to be implemented")
    def test_slice_arr_dim(self):
        pass

    def test_reverse_dim(self):
        values = np.arange(self.array_len)
        arr = xr.DataArray(values, dims=[self.dim],
                           coords={self.dim: values})
        actual = FiniteDiff(arr, self.dim)._reverse_dim()
        desired = xr.DataArray(values[::-1], dims=[self.dim],
                               coords={self.dim: values[::-1]})
        assert actual.identical(desired)

    def test_diff_not_implemented(self):
        self.assertRaises(NotImplementedError,
                          FiniteDiff(self.random, self.dim).diff)


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
            assert actual.identical(desired)

    def test_diff_zero_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self.cls(self.ones, self.dim).diff(spacing=n+1)
            desired = self.zeros_trunc[n]
            assert actual.identical(desired)

    def test_diff_const_slope_varied_arr_len(self):
        for n, arange in enumerate(self.arange_trunc[:-2]):
            actual = self.cls(arange, self.dim).diff()
            desired = self.ones_trunc[n+1]
            assert actual.identical(desired)

    def test_diff_const_slope_varied_spacing(self):
        for n, ones in enumerate(self.ones_trunc[:-1]):
            actual = self.cls(self.arange, self.dim).diff(spacing=n+1)
            desired = (n+1)*ones
            assert actual.identical(desired)

    def _compar_to_diff(self, arr):
        label = 'upper' if self.is_bwd else 'lower'
        actual = self.cls(arr, self.dim).diff()
        desired = arr.diff(self.dim, n=1, label=label)
        assert actual.identical(desired)

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
        self.method = CenDiff(self.random, self.dim).diff


@unittest.skip
class TestCenDiff(CenDiffTestCase):
    pass

if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: centered differencing
# TODO: non-default coord values (shouldn't affect diffs on arrays)
