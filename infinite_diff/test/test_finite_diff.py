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

    def test_output_coords(self):
        for n in range(self.array_len - 1):
            trunc = slice(n+1, None) if self.is_bwd else slice(0, -(n+1))
            np.testing.assert_array_equal(
                self.random[self.dim].isel(**{self.dim: trunc}),
                self.method(self.random, self.dim, spacing=n+1)[self.dim]
            )

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

    def _compar_to_diff(self, arr):
        label = 'upper' if self.is_bwd else 'lower'
        desired = arr.diff(self.dim, n=1, label=label)
        actual = self.method(arr, self.dim, spacing=1)
        assert desired.identical(actual)

    def test_various_slopes(self):
        for arr in [self.ones, self.zeros, self.arange, self.random]:
            self._compar_to_diff(arr)


class TestBwdDiff(TestFwdDiff):
    def setUp(self):
        super(TestBwdDiff, self).setUp()
        self.method = FiniteDiff.bwd_diff
        self.is_bwd = True

    def test_reverse_dim(self):
        arr = xr.DataArray(np.arange(10), dims=[self.dim],
                           coords={self.dim: np.arange(10)})
        actual = FiniteDiff.reverse_dim(arr, self.dim)
        desired = xr.DataArray(np.arange(10)[::-1], dims=[self.dim],
                               coords={self.dim: np.arange(10)[::-1]})
        assert desired.identical(actual)


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
        self.is_bwd = False


class TestFwdDiffDeriv(FwdDiffDerivTestCase):
    def test_output_coords_fill(self):
        desired = self.random.coords.to_dataset()
        for order in [1, 2]:
            ans = self.method(self.random, self.dim, coord=None, spacing=1,
                              order=order, fill_edge=True).coords.to_dataset()
            assert ans.identical(desired)

    def test_output_coords_no_fill(self):
        for order in [1, 2]:
            trunc = slice(order, None) if self.is_bwd else slice(0, -order)
            desired = self.random.coords.to_dataset().isel(**{self.dim: trunc})
            ans = self.method(self.random, self.dim, coord=None, spacing=1,
                              order=order, fill_edge=False).coords.to_dataset()
            assert ans.identical(desired)

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
            np.testing.assert_array_equal(ans, self.ones_trunc[n])

    def test_order1_spacing1_no_fill(self):
        label = 'upper' if self.is_bwd else 'lower'
        desired = (self.random.diff(self.dim, label=label) /
                   self.random[self.dim].diff(self.dim, label=label))
        actual = self.method(self.random, self.dim)
        assert desired.identical(actual)

    def test_order1_spacing1_fill(self):
        label = 'lower'
        edge_label = 'upper'
        trunc = slice(-2, None)

        interior = (self.random.diff(self.dim, label=label) /
                    self.random[self.dim].diff(self.dim, label=label))
        arr_edge = self.random.isel(**{self.dim: trunc})
        edge = (arr_edge.diff(self.dim, label=edge_label) /
                arr_edge[self.dim].diff(self.dim, label=edge_label))
        desired = xr.concat([interior, edge], dim=self.dim)
        actual = self.method(self.random, self.dim, fill_edge=True)
        assert desired.identical(actual)


class TestBwdDiffDeriv(TestFwdDiffDeriv):
    def setUp(self):
        super(TestBwdDiffDeriv, self).setUp()
        self.method = FiniteDiff.bwd_diff_deriv
        self.is_bwd = True

    def test_order1_spacing1_fill(self):
        trunc = slice(0, 2)

        interior = (FiniteDiff.bwd_diff(self.random, self.dim) /
                    FiniteDiff.bwd_diff(self.random[self.dim], self.dim))
        arr_edge = self.random.isel(**{self.dim: trunc})
        edge = (FiniteDiff.fwd_diff(arr_edge, self.dim) /
                FiniteDiff.fwd_diff(arr_edge[self.dim], self.dim))
        desired = xr.concat([edge, interior], dim=self.dim)
        actual = self.method(self.random, self.dim, fill_edge=True)
        assert desired.identical(actual)


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
        self.orders = [1, 2]
        self.fill_edges = [False, True]

    def test_upwind_advec_flow(self):
        flow = np.abs(self.random)
        _, pos = FiniteDiff.upwind_advec_flow(flow)
        assert flow.identical(pos)

        flow *= -1
        neg, _ = FiniteDiff.upwind_advec_flow(flow)
        assert flow.identical(neg)

        flow = xr.DataArray(np.random.uniform(low=-5, high=5,
                                              size=self.random.shape),
                            dims=self.random.dims, coords=self.random.coords)
        neg, pos = FiniteDiff.upwind_advec_flow(flow)
        assert flow.identical(neg + pos)

    def test_reverse_dim(self):
        flow = np.abs(self.random)
        neg, _ = FiniteDiff.upwind_advec_flow(flow, reverse_dim=True)
        assert flow.identical(neg)

        flow *= -1
        _, pos = FiniteDiff.upwind_advec_flow(flow, reverse_dim=True)
        assert flow.identical(pos)

        flow = xr.DataArray(np.random.uniform(low=-5, high=5,
                                              size=self.random.shape),
                            dims=self.random.dims, coords=self.random.coords)
        neg, pos = FiniteDiff.upwind_advec_flow(flow, reverse_dim=True)
        assert flow.identical(neg + pos)

    def test_output_coords_fill(self):
        desired = self.arange.coords.to_dataset()
        for o in [1, 2]:
            actual = self.method(self.random, self.random, self.dim, order=o,
                                 fill_edge=True).coords.to_dataset()
            assert desired.identical(actual)

    def test_output_coords_no_fill(self):
        o1 = self.arange.coords.to_dataset().isel(**{self.dim: slice(1, -1)})
        o2 = self.arange.coords.to_dataset().isel(**{self.dim: slice(2, -2)})
        for desired, o in zip([o1, o2], [1, 2]):
            actual = self.method(self.random, self.random, self.dim, order=o,
                                 fill_edge=False).coords.to_dataset()
            assert desired.identical(actual)

    def test_zero_flow(self):

        for args in itertools.product(self.arrs, [self.zeros], self.dims,
                                      self.coords, self.spacings, self.orders,
                                      self.fill_edges):
            self.assertTrue(not np.any(self.method(*args)))

    def test_pos_flow(self):
        flow = np.abs(self.random)
        # Do fill edge.
        desired = flow * FiniteDiff.bwd_diff_deriv(
            self.arange, self.dim, coord=None, spacing=1, order=1,
            fill_edge=True
        )
        actual = self.method(self.arange, flow, self.dim, coord=None,
                             spacing=1, order=1, fill_edge=True)
        assert desired.identical(actual)

        # Do not fill edge.
        desired = flow * FiniteDiff.bwd_diff_deriv(
            self.arange, self.dim, coord=None, spacing=1, order=1,
            fill_edge=False
        ).isel(**{self.dim: slice(None, -1)})
        actual = self.method(self.arange, flow, self.dim, coord=None,
                             spacing=1, order=1, fill_edge=False)
        assert desired.identical(actual)

    def test_neg_flow(self):
        flow = -1*np.abs(self.random)
        # Do fill edge.
        desired = flow * FiniteDiff.fwd_diff_deriv(
            self.arange, self.dim, coord=None, spacing=1, order=1,
            fill_edge=True
        )
        actual = self.method(self.arange, flow, self.dim, coord=None,
                             spacing=1, order=1, fill_edge=True)
        assert desired.identical(actual)

        # Do not fill edge.
        desired = flow * FiniteDiff.fwd_diff_deriv(
            self.arange, self.dim, coord=None, spacing=1, order=1,
            fill_edge=False
        ).isel(**{self.dim: slice(1, None)})
        actual = self.method(self.arange, flow, self.dim, coord=None,
                             spacing=1, order=1, fill_edge=False)
        assert desired.identical(actual)


if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: non-default coord values
# TODO: centered differencing and derivs
# TODO: non-unity spacing for derivatives and advection
# TODO: comparison of derivatives to analytical solutions, e.g. sin/cos, e^x
