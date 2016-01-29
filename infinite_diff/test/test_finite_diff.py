#! /usr/bin/env python
"""Tests"""
import sys
import unittest

import numpy as np
import xarray as xr

from infinite_diff import FiniteDiff


class FiniteDiffTestCase(unittest.TestCase):
    def setUp(self):
        self._array_len = 10
        self.dim = 'testdim'
        self.ones = xr.DataArray(np.ones(self._array_len), dims=[self.dim])
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n, None)})
                           for n in range(self._array_len)]
        self.zeros = xr.DataArray(np.zeros(self._array_len), dims=[self.dim])
        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n, None)})
                            for n in range(self._array_len)]
        self.arange = xr.DataArray(np.arange(self._array_len), dims=[self.dim])
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n, None)})
                             for n in range(self._array_len)]

    def tearDown(self):
        pass


class FwdDiffTestCase(FiniteDiffTestCase):
    def setUp(self):
        super(FwdDiffTestCase, self).setUp()
        self.method = FiniteDiff.fwd_diff


class TestFwdDiff(FwdDiffTestCase):
    def test_bad_spacing(self):
        self.assertRaises(ValueError, self.method, self.ones, self.dim,
                          **{'spacing': 0})

    def test_bad_array_len(self):
        for n in range(len(self.ones[self.dim])):
            self.assertRaises(ValueError, self.method,
                              self.ones_trunc[n], self.dim,
                              **{'spacing': self._array_len - n})
        # TODO: array len not defined, such that len(arr[dim]) throws TypeError

    def test_zero_slope(self):
        for n, zeros in enumerate(self.zeros_trunc[1:]):
            # Array len gets progressively smaller.
            ans = self.method(self.ones_trunc[n], self.dim, spacing=1)
            np.testing.assert_array_equal(ans, zeros)
            # Spacing of differencing gets progressively larger.
            ans = self.method(self.ones, self.dim, spacing=n+1)
            np.testing.assert_array_equal(ans, zeros)

    def test_constant_slope(self):
        for n, arange in enumerate(self.arange_trunc[:-1]):
            # Array len gets progressively smaller.
            ans = self.method(arange, self.dim, spacing=1)
            np.testing.assert_array_equal(ans, self.ones_trunc[n+1])
            # Spacing of differencing gets progressively larger.
            ans = self.method(self.arange, self.dim, spacing=n+1)
            np.testing.assert_array_equal(ans, (n+1)*self.ones_trunc[n+1])

if __name__ == '__main__':
    sys.exit(unittest.main())


class TestBwdDiff(TestFwdDiff):
    def setUp(self):
        super(TestBwdDiff, self).setUp()
        self.method = FiniteDiff.bwd_diff


class CenDiffTestCase(FiniteDiffTestCase):
    def setUp(self):
        super(CenDiffTestCase, self).setUp()


class TestCenDiff(CenDiffTestCase):
    def setUp(self):
        super(TestCenDiff, self).setUp()
        self.method = FiniteDiff.cen_diff

# TODO: non-constant slope for fwd/bwd
# TODO: tests of getting proper coord values for fwd/bwd
# TODO: centered differencing tests
# TODO: derivative tests
# TODO: upwind advection tests
