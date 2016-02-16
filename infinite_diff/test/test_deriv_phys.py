import sys
import unittest

import numpy as np
import xarray as xr

from infinite_diff.deriv import PhysDeriv, LonDeriv, LatDeriv, SphereEtaDeriv
from infinite_diff.deriv import LonFwdDeriv, LatFwdDeriv, EtaFwdDeriv
from infinite_diff.deriv import LonBwdDeriv, LatBwdDeriv, EtaBwdDeriv
from infinite_diff.deriv import SphereEtaFwdDeriv, SphereEtaBwdDeriv
from infinite_diff._constants import LON_STR, LAT_STR, PFULL_STR

from . import InfiniteDiffTestCase


class PhysDerivSharedTests(object):
    def test__init__(self):
        self.assertDatasetIdentical(self.deriv_obj.arr, self.arr)
        self.assertEqual(self.deriv_obj.dim, self.dim)

    def test_deriv(self):
        self.assertNotImplemented(self.deriv_obj.deriv)


class PhysDerivTestCase(InfiniteDiffTestCase):
    _DERIV_CLS = PhysDeriv
    _CYCLIC = False
    _COORD_KWARGS = dict(cyclic=_CYCLIC)

    def setUp(self):
        super(PhysDerivTestCase, self).setUp()
        self.dim = LAT_STR
        self.arr = xr.DataArray(np.random.random(self.lat.shape),
                                dims=self.dim, coords={self.dim: self.lat})
        self.deriv_obj = self._DERIV_CLS(self.arr, self.dim,
                                         **self._COORD_KWARGS)


class TestPhysDeriv(PhysDerivSharedTests, PhysDerivTestCase):
    pass


class LonDerivTestCase(PhysDerivTestCase):
    _DERIV_CLS = LonDeriv
    _CYCLIC = True

    def setUp(self):
        super(LonDerivTestCase, self).setUp()
        self.dim = LON_STR
        self.arr = xr.DataArray(np.random.random(self.lon.shape),
                                dims=self.dim, coords={self.dim: self.lon})
        self.deriv_obj = self._DERIV_CLS(self.arr, self.dim,
                                         **self._COORD_KWARGS)


class TestLonDeriv(TestPhysDeriv, LonDerivTestCase):
    def test_deriv(self):
        self.assertNotImplemented(self.deriv_obj.deriv, 1)


class LonFwdDerivTestCase(LonDerivTestCase):
    _DERIV_CLS = LonFwdDeriv

    def setUp(self):
        super(LonFwdDerivTestCase, self).setUp()
        self.arr = xr.DataArray(np.random.random(self.lon.shape),
                                dims=self.dim, coords={self.dim: self.lon})
        self.arr2 = xr.DataArray(
            np.random.random((self.lat.size, self.lon.size)),
            dims=[LAT_STR, LON_STR],
            coords={LON_STR: self.lon, LAT_STR: self.lat}
        )
        self.deriv_obj = self._DERIV_CLS(self.arr, self.dim,
                                         **self._COORD_KWARGS)


class TestLonFwdDeriv(TestLonDeriv, LonFwdDerivTestCase):
    def test_deriv(self):
        # Scalar latitude.
        actual = self.deriv_obj.deriv(0.).shape
        desired = self.deriv_obj.arr.shape
        self.assertEqual(actual, desired)
        # Array of latitudes.
        deriv_obj = self._DERIV_CLS(self.arr2, self.dim, **self._COORD_KWARGS)
        actual = deriv_obj.deriv(self.lat).shape
        desired = self.arr2.shape
        self.assertEqual(actual, desired)


class LonBwdDerivTestCase(LonFwdDerivTestCase):
    _DERIV_CLS = LonBwdDeriv


class TestLonBwdDeriv(TestLonFwdDeriv, LonBwdDerivTestCase):
    pass


class LatDerivTestCase(PhysDerivTestCase):
    _DERIV_CLS = LatDeriv
    _COORD_KWARGS = {}

    def setUp(self):
        super(LatDerivTestCase, self).setUp()
        self.dim = LAT_STR
        self.arr = xr.DataArray(np.random.random(self.lat.shape),
                                dims=self.dim, coords={self.dim: self.lat})
        self.arr2 = xr.DataArray(
            np.random.random((self.lat.size, self.lon.size)),
            dims=[LAT_STR, LON_STR],
            coords={LON_STR: self.lon, LAT_STR: self.lat}
        )
        self.deriv_obj = self._DERIV_CLS(self.arr, self.dim,
                                         **self._COORD_KWARGS)


class TestLatDeriv(PhysDerivSharedTests, LatDerivTestCase):
    def test_deriv(self):
        self.assertNotImplemented(self.deriv_obj.deriv, 'divg')


class LatFwdDerivTestCase(LatDerivTestCase):
    _DERIV_CLS = LatFwdDeriv
    _CYCLIC = False

    def setUp(self):
        super(LatFwdDerivTestCase, self).setUp()
        self.dim = LAT_STR
        self.arr = xr.DataArray(np.random.random(self.lat.shape),
                                dims=self.dim, coords={self.dim: self.lat})
        self.arr2 = xr.DataArray(
            np.random.random((self.lat.size, self.lon.size)),
            dims=[LAT_STR, LON_STR],
            coords={LON_STR: self.lon, LAT_STR: self.lat}
        )
        self.deriv_obj = self._DERIV_CLS(self.arr, self.dim,
                                         **self._COORD_KWARGS)


class TestLatFwdDeriv(TestLatDeriv, LatFwdDerivTestCase):
    def test_deriv(self):
        for oper in ['grad', 'divg']:
            # Scalar latitude.
            actual = self.deriv_obj.deriv(oper=oper).shape
            desired = self.deriv_obj.arr.shape
            self.assertEqual(actual, desired)
            # Array of latitudes.
            deriv_obj = self._DERIV_CLS(self.arr2, self.dim,
                                        **self._COORD_KWARGS)
            actual = deriv_obj.deriv(oper=oper).shape
            desired = self.arr2.shape
            self.assertEqual(actual, desired)


class LatBwdDerivTestCase(LatFwdDerivTestCase):
    _DERIV_CLS = LatBwdDeriv


class TestLatBwdDeriv(TestLatFwdDeriv, LatBwdDerivTestCase):
    pass


class EtaFwdDerivTestCase(InfiniteDiffTestCase):
    _DERIV_CLS = EtaFwdDeriv

    def setUp(self):
        super(EtaFwdDerivTestCase, self).setUp()
        self.dim = PFULL_STR
        self.arr = xr.DataArray(np.random.random(self.pfull.shape),
                                dims=self.dim, coords={self.dim: self.pfull})
        self.ps = 1e5
        self.deriv_obj = self._DERIV_CLS(self.arr, self.pk, self.bk, self.ps)


class TestEtaFwdDeriv(PhysDerivSharedTests, EtaFwdDerivTestCase):
    def test_deriv(self):
        self.deriv_obj.deriv()


class EtaBwdDerivTestCase(EtaFwdDerivTestCase):
    _DERIV_CLS = EtaBwdDeriv


class TestEtaBwdDeriv(TestEtaFwdDeriv, EtaBwdDerivTestCase):
    pass


class SphereEtaDerivTestCase(InfiniteDiffTestCase):
    _DERIV_CLS = SphereEtaDeriv

    def setUp(self):
        super(SphereEtaDerivTestCase, self).setUp()
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.ps = xr.DataArray(
            np.random.random((len(self.lat), len(self.lon))),
            dims=[LAT_STR, LON_STR],
            coords={LAT_STR: self.lat, LON_STR: self.lon}
        )*1e3 + 1e5

        self.deriv_obj = self._DERIV_CLS(self.arr, self.pk, self.bk, self.ps)


class TestSphereEtaDeriv(PhysDerivSharedTests, SphereEtaDerivTestCase):
    def test__init__(self):
        self.assertDatasetIdentical(self.deriv_obj.arr, self.arr)

    def test_deriv(self):
        self.assertNotImplemented(self.deriv_obj.d_dx)
        self.assertNotImplemented(self.deriv_obj.d_dy, 'divg')


class SphereEtaFwdDerivTestCase(SphereEtaDerivTestCase):
    _DERIV_CLS = SphereEtaFwdDeriv


class TestSphereEtaFwdDeriv(TestSphereEtaDeriv, SphereEtaFwdDerivTestCase):
    def test_deriv(self):
        self.deriv_obj.d_dx()
        self.deriv_obj.d_dy()
        self.deriv_obj.horiz_grad()

    def test_deriv_const_p(self):
        self.deriv_obj.d_dx_const_p()
        self.deriv_obj.d_dy_const_p()
        self.deriv_obj.horiz_grad_const_p()


class SphereEtaBwdDerivTestCase(SphereEtaDerivTestCase):
    _DERIV_CLS = SphereEtaBwdDeriv


class TestSphereEtaBwdDeriv(TestSphereEtaFwdDeriv, SphereEtaBwdDerivTestCase):
    pass


if __name__ == '__main__':
    sys.exit(unittest.main())
