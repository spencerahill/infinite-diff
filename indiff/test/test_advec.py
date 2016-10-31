import itertools
import sys
import unittest

import numpy as np
import xarray as xr

from indiff import (Advec, CenAdvec, Upwind, FiniteDeriv, BwdDeriv, FwdDeriv,
                    CenDeriv)

from . import InfiniteDiffTestCase


class AdvecSharedTests(object):
    def test_init(self):
        self.assertIs(self.advec_obj.flow, self.flow)
        self.assertIs(self.advec_obj.arr, self.arr)
        self.assertEqual(self.advec_obj.dim, self.dim)
        self.assertIsInstance(self.advec_obj._deriv_obj, self._DERIV_CLS)

    def test_arr_gradient(self):
        self.assertNotImplemented(self.advec_obj._arr_gradient)

    def test_advec(self):
        self.assertNotImplemented(self.advec_obj.advec)


class AdvecTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = Advec
    _DERIV_CLS = FiniteDeriv

    def setUp(self):
        super(AdvecTestCase, self).setUp()
        self.arr = self.random
        self.flow = self.random2
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self.dim)


class TestAdvec(AdvecSharedTests, AdvecTestCase):
    pass


class CenAdvecTestCase(AdvecTestCase):
    _ADVEC_CLS = CenAdvec
    _DERIV_CLS = CenDeriv


class TestCenAdvec(CenAdvecTestCase, TestAdvec):
    def setUp(self):
        super(TestCenAdvec, self).setUp()

    def test_arr_gradient(self):
        desired = self._DERIV_CLS(self.arr, self.dim, fill_edge=True).deriv()
        actual = self.advec_obj._arr_gradient()
        self.assertDatasetIdentical(actual, desired)

    def test_advec(self):
        self.advec_obj.advec()


class UpwindTestCase(AdvecTestCase):
    _ADVEC_CLS = Upwind

    def setUp(self):
        super(UpwindTestCase, self).setUp()


class TestUpwind(UpwindTestCase):
    def test_flow_neg_pos(self):
        _, pos = self.advec_obj._flow_neg_pos()
        self.assertDatasetIdentical(pos, self.flow)

        uw = self._ADVEC_CLS(-1*np.abs(self.random2), self.arr, self.dim)
        desired = -1*self.random2
        neg, _ = uw._flow_neg_pos()
        self.assertDatasetIdentical(neg, desired)

        flow = xr.DataArray(np.random.uniform(low=-5, high=5,
                                              size=self.random.shape),
                            dims=self.random.dims, coords=self.random.coords)
        uw = self._ADVEC_CLS(flow, self.arr, self.dim)
        neg, pos = uw._flow_neg_pos()
        self.assertDatasetIdentical(flow, neg + pos)

    def test_advec_output_coords_fill(self):
        desired = self.arr.coords.to_dataset()
        for o in [1, 2]:
            actual = self._ADVEC_CLS(
                self.random, self.arr, self.dim, order=o, fill_edge=True
            ).advec().coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_output_coords_no_fill(self):
        o1 = self.arr.coords.to_dataset()[{self.dim: slice(1, -1)}]
        o2 = self.arr.coords.to_dataset()[{self.dim: slice(2, -2)}]
        for desired, o in zip([o1, o2], [1, 2]):
            actual = self._ADVEC_CLS(
                self.random, self.arr, self.dim, order=o, fill_edge=False
            ).advec().coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_zero_flow(self):
        arrs = [self.random]
        dims = [self.dim, self.dummy_dim]
        coords = [None, self.random2]
        spacings = [1]
        orders = [1, 2]
        fill_edges = [False, True]
        for args in itertools.product([self.zeros], arrs, dims, coords,
                                      spacings, orders, fill_edges):
            self.assertAllZeros(self._ADVEC_CLS(*args).advec())

    def test_advec_pos_flow_fill(self):
        flow = np.abs(self.random)
        for o in [1, 2]:
            desired = flow * BwdDeriv(self.arr, self.dim, order=o,
                                      fill_edge=True).deriv()
            edge = flow * FwdDeriv(self.arr, self.dim, order=o).deriv()
            desired[{self.dim: slice(0, 1)}] = edge[{self.dim: slice(0, 1)}]
            actual = self._ADVEC_CLS(flow, self.arr, self.dim, order=o,
                                     fill_edge=True).advec()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_pos_flow_no_fill(self):
        flow = np.abs(self.random)
        for o in [1, 2]:
            desired = flow[{self.dim: slice(None, -o)}] * BwdDeriv(
                self.arr, self.dim, order=o, fill_edge=False
            ).deriv()
            actual = self._ADVEC_CLS(flow, self.arr, self.dim, order=o,
                                     fill_edge=False).advec()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_neg_flow_fill(self):
        flow = -1*np.abs(self.random)
        for o in [1, 2]:
            desired = flow * FwdDeriv(self.arr, self.dim, order=o,
                                      fill_edge=True).deriv()
            edge = flow * BwdDeriv(self.arr, self.dim, order=o).deriv()
            desired[{self.dim: slice(-1, None)}] = edge[{self.dim:
                                                         slice(-1, None)}]
            actual = self._ADVEC_CLS(flow, self.arr, self.dim, order=o,
                                     fill_edge=True).advec()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_neg_flow_no_fill(self):
        flow = -1*np.abs(self.random)
        for o in [1, 2]:
            desired = flow[{self.dim: slice(o, None)}] * FwdDeriv(
                self.arr, self.dim, order=o, fill_edge=False
            ).deriv()
            actual = self._ADVEC_CLS(flow, self.arr, self.dim, order=o,
                                     fill_edge=False).advec()
            self.assertDatasetIdentical(actual, desired)

if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: non-unity spacing
