import itertools
import sys
import unittest

import numpy as np
import xarray as xr

from infinite_diff import Advec, CenAdvec, Upwind
from infinite_diff import FiniteDeriv, BwdDeriv, FwdDeriv, CenDeriv

from . import InfiniteDiffTestCase


class AdvecTestCase(InfiniteDiffTestCase):
    def setUp(self):
        super(AdvecTestCase, self).setUp()
        self.arr = self.random
        self.flow = self.random2
        self.advec_cls = Advec
        self.deriv_cls = FiniteDeriv
        self.advec_obj = self.advec_cls(self.flow, self.arr, self.dim)
        self.method = self.advec_obj.advec


class TestAdvec(AdvecTestCase):
    def setUp(self):
        super(TestAdvec, self).setUp()

    def test_init(self):
        self.assertIs(self.advec_obj.flow, self.flow)
        self.assertIs(self.advec_obj.arr, self.arr)
        self.assertEqual(self.advec_obj.dim, self.dim)
        self.assertIsInstance(self.advec_obj._deriv_obj, self.deriv_cls)

    def test_arr_gradient(self):
        self.assertNotImplemented(self.advec_obj._arr_gradient)

    def test_advec(self):
        self.assertNotImplemented(self.method)


class CenAdvecTestCase(AdvecTestCase):
    def setUp(self):
        super(CenAdvecTestCase, self).setUp()
        self.advec_cls = CenAdvec
        self.deriv_cls = CenDeriv
        self.advec_obj = self.advec_cls(self.flow, self.arr, self.dim)
        self.method = self.advec_obj.advec


class TestCenAdvec(CenAdvecTestCase, TestAdvec):
    def setUp(self):
        super(TestCenAdvec, self).setUp()

    def test_arr_gradient(self):
        desired = self.deriv_cls(self.arr, self.dim).deriv(fill_edge=True)
        actual = self.advec_obj._arr_gradient()
        self.assertDatasetIdentical(actual, desired)

    @unittest.skip("To be implemented")
    def test_advec(self):
        pass

    # def test_advec_zero_flow(self):
        # pass


class UpwindTestCase(AdvecTestCase):
    def setUp(self):
        super(UpwindTestCase, self).setUp()
        self.advec_cls = Upwind
        self.advec_obj = self.advec_cls(self.flow, self.arr, self.dim)
        self.method = self.advec_obj.advec


class TestUpwind(UpwindTestCase):
    def setUp(self):
        super(TestUpwind, self).setUp()
        self.arrs = [self.random]
        self.dims = [self.dim, self.dummy_dim]
        self.flows = [self.zeros]
        self.coords = [None, self.random2]
        self.spacings = [1]
        self.orders = [1, 2]
        self.fill_edges = [False, True]

    def test_flow_neg_pos(self):
        _, pos = self.advec_obj._flow_neg_pos()
        self.assertDatasetIdentical(pos, self.flow)

        uw = self.advec_cls(-1*np.abs(self.random2), self.arr, self.dim)
        desired = -1*self.random2
        neg, _ = uw._flow_neg_pos()
        self.assertDatasetIdentical(neg, desired)

        flow = xr.DataArray(np.random.uniform(low=-5, high=5,
                                              size=self.random.shape),
                            dims=self.random.dims, coords=self.random.coords)
        uw = self.advec_cls(flow, self.arr, self.dim)
        neg, pos = uw._flow_neg_pos()
        self.assertDatasetIdentical(flow, neg + pos)

    # def test_derivs_bwd_fwd(self):
    #     bwd, fwd = self.advec_obj._derivs_bwd_fwd()

    # def test_reverse_dim(self):
    #     flow = np.abs(self.random)
    #     neg, _ = self.advec_obj._flow_neg_pos(flow, reverse_dim=True)
    #     self.assertDatasetIdentical(flow, neg)

    #     flow *= -1
    #     _, pos = self.advec_obj._flow_neg_pos(flow, reverse_dim=True)
    #     self.assertDatasetIdentical(flow, pos)

    #     flow = xr.DataArray(np.random.uniform(low=-5, high=5,
    #                                           size=self.random.shape),
    #                         dims=self.random.dims, coords=self.random.coords)
    #     neg, pos = self.advec_obj._flow_neg_pos(flow, reverse_dim=True)
    #     self.assertDatasetIdentical(flow, neg + pos)

    def test_advec_output_coords_fill(self):
        desired = self.arr.coords.to_dataset()
        for o in [1, 2]:
            actual = self.advec_cls(
                self.random, self.arr, self.dim, order=o, fill_edge=True
            ).advec().coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_output_coords_no_fill(self):
        o1 = self.arr.coords.to_dataset().isel(**{self.dim: slice(1, -1)})
        o2 = self.arr.coords.to_dataset().isel(**{self.dim: slice(2, -2)})
        for desired, o in zip([o1, o2], [1, 2]):
            actual = self.advec_cls(
                self.random, self.arr, self.dim, order=o, fill_edge=False
            ).advec().coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_advec_zero_flow(self):
        for args in itertools.product([self.zeros], self.arrs, self.dims,
                                      self.coords, self.spacings, self.orders,
                                      self.fill_edges):
            self.assertTrue(not np.any(self.advec_cls(*args).advec()))

    def test_pos_flow(self):
        flow = np.abs(self.random)
        # Do fill edge.
        desired = flow * BwdDeriv(self.arr, self.dim).deriv(order=1,
                                                            fill_edge=True)
        actual = self.advec_cls(flow, self.arr, self.dim, order=1,
                                fill_edge=True).advec()
        self.assertDatasetIdentical(actual, desired)

        # Do not fill edge.
        desired = flow * BwdDeriv(self.arr, self.dim).deriv(order=1,
                                                            fill_edge=False)
        desired = desired.isel(**{self.dim: slice(None, -1)})
        actual = self.advec_cls(flow, self.arr, self.dim, order=1,
                                fill_edge=False).advec()
        self.assertDatasetIdentical(actual, desired)

    def test_neg_flow(self):
        flow = -1*np.abs(self.random)
        # Do fill edge.
        desired = flow * FwdDeriv(self.arr, self.dim).deriv(order=1,
                                                            fill_edge=True)
        actual = self.advec_cls(flow, self.arr, self.dim, order=1,
                                fill_edge=True).advec()
        self.assertDatasetIdentical(actual, desired)

        # Do not fill edge.
        desired = flow * FwdDeriv(self.arr, self.dim).deriv(order=1,
                                                            fill_edge=False)
        desired = desired.isel(**{self.dim: slice(1, None)})
        actual = self.advec_cls(flow, self.arr, self.dim, order=1,
                                fill_edge=False).advec()
        self.assertDatasetIdentical(actual, desired)

if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: non-unity spacing
