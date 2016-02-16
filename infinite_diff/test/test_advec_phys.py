import sys
import unittest

import numpy as np
import xarray as xr

from infinite_diff._constants import LAT_STR, LON_STR, PFULL_STR
from infinite_diff.advec import (PhysUpwind, LonUpwind, LatUpwind, EtaUpwind,
                                 SphereEtaUpwind, LonUpwindConstP)
from infinite_diff.deriv import (PhysDeriv, LonFwdDeriv, LonBwdDeriv,
                                 LatFwdDeriv, LatBwdDeriv,
                                 EtaFwdDeriv, EtaBwdDeriv,
                                 SphereEtaFwdDeriv, SphereEtaBwdDeriv)
from . import InfiniteDiffTestCase


class PhysAdvecSharedTests(object):
    def test_init(self):
        self.assertIsInstance(self.advec_obj._deriv_bwd_obj,
                              self._DERIV_BWD_CLS)
        self.assertIsInstance(self.advec_obj._deriv_fwd_obj,
                              self._DERIV_FWD_CLS)

    def test_advec(self):
        self.assertNotImplemented(self.advec_obj.advec)


class PhysUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = PhysUpwind
    _DERIV_FWD_CLS = PhysDeriv
    _DERIV_BWD_CLS = PhysDeriv
    _DIM = LAT_STR

    def setUp(self):
        super(PhysUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
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
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self._DIM)


class TestPhysUpwind(PhysAdvecSharedTests, PhysUpwindTestCase):
    pass


class LonUpwindTestCase(PhysUpwindTestCase):
    _ADVEC_CLS = LonUpwind
    _DERIV_FWD_CLS = LonFwdDeriv
    _DERIV_BWD_CLS = LonBwdDeriv
    _DIM = LON_STR


class TestLonUpwind(PhysAdvecSharedTests, LonUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec(self.lat)


class LatUpwindTestCase(LonUpwindTestCase):
    _ADVEC_CLS = LatUpwind
    _DERIV_FWD_CLS = LatFwdDeriv
    _DERIV_BWD_CLS = LatBwdDeriv
    _DIM = LAT_STR


class TestLatUpwind(PhysAdvecSharedTests, LatUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec()


class EtaUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = EtaUpwind
    _DERIV_FWD_CLS = EtaFwdDeriv
    _DERIV_BWD_CLS = EtaBwdDeriv

    def setUp(self):
        super(EtaUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
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
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self.pk, self.bk,
                                         self.ps)


class TestEtaUpwind(PhysAdvecSharedTests, EtaUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec()


class LonUpwindConstPTestCase(EtaUpwindTestCase):
    _ADVEC_CLS = LonUpwindConstP
    _DERIV_FWD_CLS = SphereEtaFwdDeriv
    _DERIV_BWD_CLS = SphereEtaBwdDeriv

    def setUp(self):
        super(LonUpwindConstPTestCase, self).setUp()


class TestLonUpwindConstP(PhysAdvecSharedTests, LonUpwindConstPTestCase):
    def test_advec(self):
        self.advec_obj.advec()


class SphereEtaUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = SphereEtaUpwind

    def setUp(self):
        super(SphereEtaUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
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
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.advec_obj = self._ADVEC_CLS(self.arr, self.pk, self.bk, self.ps)


class TestSphereEtaUpwind(SphereEtaUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec_x_const_p(self.flow)
        self.advec_obj.advec_y_const_p(self.flow)
        self.advec_obj.advec_p(self.flow)


if __name__ == '__main__':
    sys.exit(unittest.main())
