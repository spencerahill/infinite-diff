import xarray as xr

from infinite_diff import FiniteDiff, OneSidedDiff, BwdDiff, FwdDiff, CenDiff
from infinite_diff import (FiniteDeriv, OneSidedDeriv, BwdDeriv,
                           FwdDeriv, CenDeriv)

from . import InfiniteDiffTestCase


class FiniteDerivTestCase(InfiniteDiffTestCase):
    def setUp(self):
        super(FiniteDerivTestCase, self).setUp()
        self.diff_cls = FiniteDiff
        self.deriv_cls = FiniteDeriv


class TestFiniteDeriv(FiniteDerivTestCase):
    def setUp(self):
        super(TestFiniteDeriv, self).setUp()
        self.arr = self.random
        self.deriv_obj = self.deriv_cls(self.arr, self.dim)

    def test_arr_coord(self):
        self.assertDatasetIdentical(self.deriv_obj._arr_coord(None),
                                    self.arr[self.dim])
        self.assertDatasetIdentical(self.deriv_obj._arr_coord(self.random),
                                    self.random)

    def test_init(self):
        assert self.arr.identical(self.deriv_obj.arr)
        self.assertEqual(self.dim, self.deriv_obj.dim)
        assert self.arr[self.dim].identical(self.deriv_obj._arr_coord(None))


class TestOneSidedDeriv(TestFiniteDeriv):
    def setUp(self):
        super(TestOneSidedDeriv, self).setUp()
        self.diff_cls = OneSidedDiff
        self.deriv_cls = OneSidedDeriv
        self.arr = self.random
        self.deriv_obj = self.deriv_cls(self.arr, self.dim)

    def test_init(self):
        self.assertIs(self.deriv_obj.arr, self.arr)
        self.assertEqual(self.deriv_obj.dim, self.dim)
        self.assertIsInstance(self.deriv_obj._fin_diff_obj, OneSidedDiff)

    def test_slice_edge(self):
        self.assertRaises(NotImplementedError, self.deriv_obj._slice_edge,
                          self.arr, 1, 1)

    def test_edge_deriv_rev(self):
        self.assertRaises(NotImplementedError,
                          self.deriv_obj._edge_deriv_rev, 1, 1)

    def test_concat(self):
        self.assertRaises(NotImplementedError, self.deriv_obj._concat)

    def test_private_deriv(self):
        self.assertRaises(NotImplementedError, self.deriv_obj.deriv)

    def test_public_deriv(self):
        self.assertRaises(NotImplementedError, self.deriv_obj._deriv)


class FwdDerivTestCase(FiniteDerivTestCase):
    def setUp(self):
        super(FwdDerivTestCase, self).setUp()
        self.diff_cls = FwdDiff
        self.deriv_cls = FwdDeriv


class TestFwdDeriv(FwdDerivTestCase):
    def setUp(self):
        super(TestFwdDeriv, self).setUp()
        self.arr = self.arange
        self.deriv_obj = self.deriv_cls(self.arr, self.dim)
        self.method = self.deriv_obj.deriv
        self.is_bwd = False

    def test_slice_edge(self):
        spacing, order, pad = 1, 1, 1
        actual = self.deriv_obj._slice_edge(self.arr, spacing, order)
        desired = self.arr.isel(**{self.dim:
                                   slice(-(spacing*order + pad), None)})
        self.assertDatasetIdentical(actual, desired)

    def test_diff_arr(self):
        desired = self.diff_cls(self.arr, self.dim).diff()
        actual = self.deriv_obj._diff()
        self.assertDatasetIdentical(actual, desired)

    def test_diff_coord(self):
        desired = self.diff_cls(self.arr[self.dim], self.dim).diff()
        actual = self.deriv_obj._diff(arr=self.deriv_obj.arr[self.dim])
        self.assertDatasetIdentical(actual, desired)

    def test_concat(self):
        actual = self.deriv_obj._concat(self.ones, self.arr)
        desired = xr.concat([self.ones, self.arr], dim=self.dim)
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_output_coords_fill(self):
        desired = self.random.coords.to_dataset()
        for o in [1, 2]:
            actual = self.method(order=o, fill_edge=True).coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_deriv_output_coords_no_fill(self):
        for o in [1, 2]:
            trunc = slice(o, None) if self.is_bwd else slice(0, -o)
            desired = self.random.coords.to_dataset().isel(**{self.dim: trunc})
            actual = self.method(order=o, fill_edge=False).coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order1(self):
        actual = self.method(fill_edge=False)
        desired = self.ones_trunc[0]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order1_fill(self):
        actual = self.method(fill_edge=True)
        desired = self.ones
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order2(self):
        actual = self.method(order=2, fill_edge=False)
        desired = self.ones_trunc[1]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order2_fill(self):
        actual = self.method(order=2, fill_edge=True)
        desired = self.ones
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order1(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=1,
                                                           fill_edge=False)
        desired = self.zeros_trunc[0]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order1_fill(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=1,
                                                           fill_edge=True)
        desired = self.zeros
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order2(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=2,
                                                           fill_edge=False)
        desired = self.zeros_trunc[1]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order2_fill(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=2,
                                                           fill_edge=True)
        desired = self.zeros
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_order1_spacing1_no_fill(self):
        label = 'upper' if self.is_bwd else 'lower'
        desired = (self.random.diff(self.dim, label=label) /
                   self.random[self.dim].diff(self.dim, label=label))
        actual = self.deriv_cls(self.random, self.dim).deriv()
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_order1_spacing1_fill(self):
        label = 'lower'
        edge_label = 'upper'
        trunc = slice(-2, None)

        interior = (self.random.diff(self.dim, label=label) /
                    self.random[self.dim].diff(self.dim, label=label))

        arr_edge = self.random.isel(**{self.dim: trunc})
        edge = (arr_edge.diff(self.dim, label=edge_label) /
                arr_edge[self.dim].diff(self.dim, label=edge_label))

        desired = xr.concat([interior, edge], dim=self.dim)
        actual = self.deriv_cls(self.random, self.dim).deriv(fill_edge=True)
        self.assertDatasetIdentical(actual, desired)


class TestBwdDeriv(TestFwdDeriv):
    def setUp(self):
        super(TestBwdDeriv, self).setUp()
        self.arr = self.arange
        self.diff_cls = BwdDiff
        self.deriv_cls = BwdDeriv
        self.deriv_obj = self.deriv_cls(self.arr, self.dim)
        self.method = self.deriv_obj.deriv
        self.is_bwd = True

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, None)})
                            for n in range(self.array_len)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, None)})
                           for n in range(self.array_len)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]

    def test_slice_edge(self):
        spacing, order, pad = 1, 1, 1
        actual = self.deriv_obj._slice_edge(self.arr, spacing, order)
        desired = self.arr.isel(**{self.dim:
                                   slice(None, (spacing*order + pad))})
        self.assertDatasetIdentical(actual, desired)

    def test_concat(self):
        actual = self.deriv_obj._concat(self.ones, self.arr)
        desired = xr.concat([self.arr, self.ones], dim=self.dim)
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_order1_spacing1_fill(self):
        trunc = slice(0, 2)

        interior = (self.diff_cls(self.random, self.dim).diff() /
                    self.diff_cls(self.random[self.dim], self.dim).diff())

        arr_edge = self.random.isel(**{self.dim: trunc})
        edge = (self.diff_cls(arr_edge, self.dim).diff_rev() /
                self.diff_cls(arr_edge[self.dim], self.dim).diff_rev())

        desired = xr.concat([edge, interior], dim=self.dim)
        actual = self.deriv_cls(self.random, self.dim).deriv(fill_edge=True)
        self.assertDatasetIdentical(actual, desired)


class CenDerivTestCase(FiniteDerivTestCase):
    def setUp(self):
        super(CenDerivTestCase, self).setUp()
        self.arr = self.arange
        self.diff_cls = CenDiff
        self.deriv_cls = CenDeriv
        self.deriv_obj = self.deriv_cls(self.arr, self.dim)
        self.method = self.deriv_obj.deriv


class TestCenDeriv(CenDerivTestCase):
    def setUp(self):
        super(TestCenDeriv, self).setUp()

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, -(n+1))})
                            for n in range(self.array_len // 2 - 1)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, -(n+1))})
                           for n in range(self.array_len // 2 - 1)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]

    def test_init(self):
        assert self.arr.identical(self.deriv_obj.arr)
        self.assertEqual(self.dim, self.deriv_obj.dim)
        assert self.arr[self.dim].identical(self.deriv_obj._arr_coord(None))
        self.assertIsInstance(self.deriv_obj._fin_diff_obj, CenDiff)
        self.assertIsInstance(self.deriv_obj._deriv_fwd_obj, FwdDeriv)
        self.assertIsInstance(self.deriv_obj._deriv_bwd_obj, BwdDeriv)

    def test_concat(self):
        actual = self.deriv_obj._concat(self.random, self.ones, self.arr)
        desired = xr.concat([self.random, self.ones, self.arr], dim=self.dim)
        self.assertDatasetIdentical(actual, desired)

    def test_diff_arr(self):
        desired = self.diff_cls(self.arr, self.dim).diff()
        actual = self.deriv_obj._diff()
        self.assertDatasetIdentical(actual, desired)

    def test_diff_coord(self):
        desired = self.diff_cls(self.arr[self.dim], self.dim).diff()
        actual = self.deriv_obj._diff(arr=self.deriv_obj.arr[self.dim])
        self.assertDatasetIdentical(actual, desired)

    def test_private_deriv(self):
        desired = (CenDiff(self.zeros, self.dim).diff() /
                   CenDiff(self.zeros[self.dim], self.dim).diff())
        actual = self.deriv_cls(self.zeros, self.dim)._deriv()
        self.assertDatasetIdentical(actual, desired)

    def test_output_coords_no_fill(self):
        for o in [1, 2]:
            trunc = slice(o, -o)
            desired = self.random.coords.to_dataset().isel(**{self.dim: trunc})
            actual = self.method(order=2*o,
                                 fill_edge=False).coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_output_coords_fill(self):
        desired = self.random.coords.to_dataset()
        for o in [2, 4]:
            actual = self.method(order=o, fill_edge=True).coords.to_dataset()
            self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order2(self):
        actual = self.method(fill_edge=False)
        desired = self.ones_trunc[0]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order2_fill(self):
        actual = self.method(fill_edge=True)
        desired = self.ones
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order4(self):
        actual = self.method(order=4, fill_edge=False)
        desired = self.ones_trunc[1]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_constant_slope_order4_fill(self):
        actual = self.method(order=4, fill_edge=True)
        desired = self.ones
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order2(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=2,
                                                           fill_edge=False)
        desired = self.zeros_trunc[0]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order2_fill(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=2,
                                                           fill_edge=True)
        desired = self.zeros
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order4(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=4,
                                                           fill_edge=False)
        desired = self.zeros_trunc[1]
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_zero_slope_order4_fill(self):
        actual = self.deriv_cls(self.ones, self.dim).deriv(order=4,
                                                           fill_edge=True)
        desired = self.zeros
        self.assertDatasetIdentical(actual, desired)

    def test_deriv_order2_spacing1_fill(self):
        interior = (self.diff_cls(self.random, self.dim).diff() /
                    self.diff_cls(self.random[self.dim], self.dim).diff())

        trunc_left = slice(0, 2)
        arr_left = self.random.isel(**{self.dim: trunc_left})
        left = FwdDeriv(arr_left, self.dim).deriv(order=1)

        trunc_right = slice(-2, None)
        arr_right = self.random.isel(**{self.dim: trunc_right})
        right = BwdDeriv(arr_right, self.dim).deriv(order=1)

        desired = xr.concat([left, interior, right], dim=self.dim)
        actual = self.deriv_cls(self.random, self.dim).deriv(fill_edge=True)
        self.assertDatasetIdentical(actual, desired)

# TODO: non-unity spacing
# TODO: comparison to analytical solutions, e.g. sin/cos, e^x
