#! /usr/bin/env python
"""Tests"""
import itertools
import sys
import unittest

from infinite_diff import FiniteDiff
import numpy as np
import xarray as xr

from . import TestCase


class FiniteDiffTestCase(TestCase):
    def setUp(self):
        self.array_len = 10
        self.dim = 'testdim'
        self.dummy_len = 3
        self.dummy_dim = 'dummydim'
        self.ones = xr.DataArray(np.ones((self.dummy_len, self.array_len)),
                                 dims=[self.dummy_dim, self.dim])
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n, None)})
                           for n in range(self.array_len)]
        self.zeros = xr.DataArray(np.zeros(self.ones.shape),
                                  dims=self.ones.dims)
        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n, None)})
                            for n in range(self.array_len)]
        self.arange = xr.DataArray(
            np.arange(self.array_len*self.dummy_len).reshape(self.ones.shape),
            dims=self.ones.dims
        )
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n, None)})
                             for n in range(self.array_len)]
        randstate = np.random.RandomState(12345)
        self.random = xr.DataArray(randstate.rand(*self.ones.shape),
                                   dims=self.ones.dims)

    def tearDown(self):
        pass


class FwdDiffTestCase(FiniteDiffTestCase):
    def setUp(self):
        super(FwdDiffTestCase, self).setUp()
        self.method = FiniteDiff.fwd_diff
        self.is_bwd = False


class TestFwdDiff(FwdDiffTestCase):
    def test_bad_spacing(self):
        self.assertRaises(ValueError, self.method, self.ones, self.dim,
                          **{'spacing': 0})
        self.assertRaises(TypeError, self.method, self.ones, self.dim,
                          **{'spacing': 1.1})

    def test_bad_array_len(self):
        for n in range(len(self.ones[self.dim])):
            self.assertRaises(ValueError, self.method, self.ones_trunc[n],
                              self.dim, **{'spacing': self.array_len - n})
            self.assertRaises(ValueError, self.method,
                              self.ones.isel(**{self.dim: 0}), self.dim)

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

    def test_output_coords(self):
        for n in range(self.array_len - 1):
            trunc = slice(n+1, None) if self.is_bwd else slice(0, -(n+1))
            np.testing.assert_array_equal(
                self.random[self.dim].isel(**{self.dim: trunc}),
                self.method(self.random, self.dim, spacing=n+1)[self.dim]
            )

    def test_varying_slope(self):
        label = 'upper' if self.is_bwd else 'lower'
        desired = self.random.diff(self.dim, n=1, label=label)
        actual = self.method(self.random, self.dim, spacing=1)
        np.testing.assert_array_equal(desired, actual)


class TestBwdDiff(TestFwdDiff):
    def setUp(self):
        super(TestBwdDiff, self).setUp()
        self.method = FiniteDiff.bwd_diff
        self.is_bwd = True


class CenDiffTestCase(FiniteDiffTestCase):
    def setUp(self):
        super(CenDiffTestCase, self).setUp()
        self.method = FiniteDiff.cen_diff


class TestCenDiff(CenDiffTestCase):
    def test_bad_array_len(self):
        self.assertRaises(ValueError, self.method, self.ones,
                          self.dim, **{'spacing': 5})
        self.assertRaises(ValueError, self.method,
                          self.ones.isel(**{self.dim: 0}), self.dim)


class FwdDiffDerivTestCase(FiniteDiffTestCase):
    def setUp(self):
        super(FwdDiffDerivTestCase, self).setUp()
        self.method = FiniteDiff.fwd_diff_deriv


class TestFwdDiffDeriv(FwdDiffDerivTestCase):
    def test_constant_slope(self, order=1):
        for n, arange in enumerate(self.arange_trunc[:-1]):
            # Array len gets progressively smaller.
            ans = self.method(arange, self.dim, coord=None, spacing=1,
                              order=order, fill_edge=False)
            np.testing.assert_array_equal(ans, self.ones_trunc[n+order])
            # Same, with fill_edge=True.
            ans = self.method(arange, self.dim, coord=None, spacing=1,
                              order=order, fill_edge=True)
            np.testing.assert_array_equal(ans, self.ones_trunc[n])
            # Spacing of differencing gets progressively larger.
            ans = self.method(self.arange, self.dim, coord=None, spacing=n+1,
                              order=order, fill_edge=False)
            np.testing.assert_array_equal(ans, self.ones_trunc[n+order])

    def test_constant_slope_order2(self):
        for n, arange in enumerate(self.arange_trunc[:-2]):
            # Array len gets progressively smaller.
            ans = self.method(arange, self.dim, coord=None, spacing=1, order=2,
                              fill_edge=False)
            np.testing.assert_array_equal(ans, self.ones_trunc[n+2])
            # Same, with fill_edge=True.
            ans = self.method(arange, self.dim, coord=None, spacing=1, order=2,
                              fill_edge=True)
            np.testing.assert_array_equal(ans, self.ones_trunc[n+1])


class TestBwdDiffDeriv(TestFwdDiffDeriv):
    def setUp(self):
        super(TestBwdDiffDeriv, self).setUp()
        self.method = FiniteDiff.bwd_diff_deriv

    @unittest.skip("Needs to be implemented")
    def test_reverse_dim(self):
        pass

    @unittest.skip("Needs to be implemented")
    def test_coord_reverse(self):
        pass


class UpwindAdvecTestCase(FiniteDiffTestCase):
    def setUp(self):
        super(UpwindAdvecTestCase, self).setUp()
        self.method = FiniteDiff.upwind_advec


class TestUpwindAdvec(UpwindAdvecTestCase):
    def setUp(self):
        super(TestUpwindAdvec, self).setUp()
        self.arrs = [self.arange, self.ones, self.zeros, self.random]
        self.dims = [self.dim]
        self.flows = [self.zeros]
        self.coords = [None]
        self.spacings = [1]
        self.orders = [1]
        self.fill_edges = [False, True]

    def test_zero_flow(self):
        for args in itertools.product(self.arrs, [self.zeros], self.dims,
                                      self.coords, self.spacings, self.orders,
                                      self.fill_edges):
            self.assertTrue(not np.any(self.method(*args)))

    def test_unidirectional_flow(self):
        flow = np.abs(self.random)
        np.testing.assert_array_equal(
            flow * FiniteDiff.bwd_diff_deriv(
                self.arange, self.dim, coord=None, spacing=1,
                order=1, fill_edge=False
            ).isel(**{self.dim: slice(1, None)}),
            self.method(self.arange, flow, self.dim, coord=None,
                        spacing=1, order=1, fill_edge=False)
        )
        flow *= -1
        np.testing.assert_array_equal(
            flow * FiniteDiff.fwd_diff_deriv(
                self.arange, self.dim, coord=None, spacing=1,
                order=1, fill_edge=False
            ).isel(**{self.dim: slice(1, None)}),
            self.method(self.arange, flow, self.dim, coord=None,
                        spacing=1, order=1, fill_edge=False)
        )


if __name__ == '__main__':
    sys.exit(unittest.main())


# TODO: non-constant slope for fwd/bwd
# TODO: centered differencing tests
# TODO: derivative tests
# TODO: upwind advection tests
