import sys
import unittest


import numpy as np

from indiff._constants import LON_STR, LAT_STR, PHALF_STR
from indiff.utils import to_radians
from indiff.coord import (Coord, HorizCoord, XCoord, YCoord, Lon, Lat,
                          VertCoord, ZCoord, Pressure, Sigma, Eta)

from . import InfiniteDiffTestCase


class CoordSharedTests(object):
    def test_init(self):
        self.assertIsInstance(self.coord_obj, self._COORD_CLS)
        self.assertDatasetIdentical(self.coord_obj.arr, self.arr)

    def test_getitem(self):
        for key in range(len(self.arr)):
            self.assertDatasetIdentical(self.coord_obj[key],
                                        self.coord_obj.arr[key])


class CoordTestCase(InfiniteDiffTestCase):
    _COORD_CLS = Coord
    _CYCLIC = False
    _INIT_KWARGS = dict(cyclic=_CYCLIC)

    def setUp(self):
        super(CoordTestCase, self).setUp()
        self.arr = self.arange
        self.coord_obj = Coord(self.arr, dim=self.dim, **self._INIT_KWARGS)


class TestCoord(CoordSharedTests, CoordTestCase):
    def test_deriv_prefactor(self):
        self.assertNotImplemented(self.coord_obj.deriv_prefactor)

    def test_deriv_factor(self):
        self.assertNotImplemented(self.coord_obj.deriv_factor)


class HorizCoordTestCase(CoordTestCase):
    _COORD_CLS = HorizCoord

    def setUp(self):
        super(HorizCoordTestCase, self).setUp()
        self.coord_obj = HorizCoord(self.arr, dim=self.dim,
                                    cyclic=self._CYCLIC)


class TestHorizCoord(TestCoord, HorizCoordTestCase):
    pass


class XCoordTestCase(HorizCoordTestCase):
    _COORD_CLS = XCoord

    def setUp(self):
        super(XCoordTestCase, self).setUp()
        self.coord_obj = XCoord(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class TestXCoord(XCoordTestCase, TestHorizCoord):
    def test_deriv_prefactor(self):
        self.assertEqual(self.coord_obj.deriv_prefactor(), 1.)

    def test_deriv_factor(self):
        self.assertEqual(self.coord_obj.deriv_factor(), 1.)


class YCoordTestCase(XCoordTestCase):
    _COORD_CLS = YCoord

    def setUp(self):
        super(YCoordTestCase, self).setUp()


class TestYCoord(TestXCoord, YCoordTestCase):
    def setUp(self):
        super(TestYCoord, self).setUp()
        self.coord_obj = YCoord(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class LonTestCase(XCoordTestCase):
    _COORD_CLS = Lon
    _CYCLIC = True

    def setUp(self):
        super(LonTestCase, self).setUp()
        self.arr = self.lon
        self.dim = LON_STR
        self.coord_obj = Lon(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class TestLon(LonTestCase, TestXCoord):
    def test_deriv_prefactor(self):
        desired = 1. / (self.coord_obj.radius * np.cos(to_radians(self.lat)))
        actual = self.coord_obj.deriv_prefactor(self.lat)
        self.assertDatasetIdentical(actual, desired)


class LatTestCase(YCoordTestCase):
    _COORD_CLS = Lat
    _INIT_KWARGS = {}

    def setUp(self):
        super(LatTestCase, self).setUp()
        self.arr = self.lat
        self.dim = LAT_STR
        self.coord_obj = Lat(self.arr, dim=self.dim)


class TestLat(LatTestCase, TestYCoord):
    def test_deriv_prefactor(self):
        # Gradient
        desired = 1. / self.coord_obj.radius
        actual = self.coord_obj.deriv_prefactor()
        self.assertEqual(actual, desired)
        # Divergence
        desired = 1. / (self.coord_obj.radius*np.cos(self.coord_obj._lat_rad))
        actual = self.coord_obj.deriv_prefactor(oper='divg')
        self.assertDatasetIdentical(actual, desired)
        # Invalid
        self.assertRaises(ValueError, self.coord_obj.deriv_prefactor, 'abc')

    def test_deriv_factor(self):
        # Gradient
        desired = 1.
        actual = self.coord_obj.deriv_factor()
        self.assertEqual(actual, desired)
        # Divergence
        desired = np.cos(self.coord_obj._lat_rad)
        actual = self.coord_obj.deriv_factor(oper='divg')
        self.assertDatasetIdentical(actual, desired)
        # Invalid
        self.assertRaises(ValueError, self.coord_obj.deriv_factor, 'abc')


class VertCoordTestCase(CoordTestCase):
    _COORD_CLS = VertCoord
    _CYCLIC = False

    def setUp(self):
        super(VertCoordTestCase, self).setUp()
        self.coord_obj = VertCoord(self.arr, dim=self.dim)


class TestVertCoord(VertCoordTestCase, TestCoord):
    pass


class ZCoordTestCase(VertCoordTestCase):
    _COORD_CLS = ZCoord

    def setUp(self):
        super(ZCoordTestCase, self).setUp()
        self.coord_obj = ZCoord(self.arr, dim=self.dim)


class TestZCoord(ZCoordTestCase, TestVertCoord):
    pass


class PressureTestCase(VertCoordTestCase):
    _COORD_CLS = Pressure

    def setUp(self):
        super(PressureTestCase, self).setUp()
        self.arr = self.pressure
        self.dim = 'pressure'
        self.coord_obj = Pressure(self.arr, dim=self.dim)


class TestPressure(PressureTestCase, TestVertCoord):
    pass


class SigmaTestCase(VertCoordTestCase):
    _COORD_CLS = Sigma

    def setUp(self):
        super(SigmaTestCase, self).setUp()
        self.arr = self.sigma
        self.dim = 'sigma'
        self.coord_obj = Sigma(self.arr, dim=self.dim)


class TestSigma(SigmaTestCase, TestVertCoord):
    pass


class EtaTestCase(VertCoordTestCase):
    _COORD_CLS = Eta

    def setUp(self):
        super(EtaTestCase, self).setUp()
        self.arr = self.phalf
        self.ps = self.ones.copy()
        self.ps.values = 1e5 + 1e3*np.random.random(self.ones.shape)
        self.dim = PHALF_STR
        self.coord_obj = Eta(self.pk, self.bk, self.pfull, dim=self.dim)


class TestEta(EtaTestCase, TestVertCoord):
    def test_init(self):
        self.assertIsInstance(self.coord_obj, self._COORD_CLS)
        self.assertCoordsIdentical(self.coord_obj.arr, self.arr)

    def test_phalf_from_ps(self):
        for ps in [1e5, self.ps]:
            actual = self.coord_obj.phalf_from_ps(ps)
            desired = self.pk + self.bk*ps
            self.assertDatasetIdentical(actual, desired)

    def test_to_pfull_from_phalf(self):
        actual = self.coord_obj.to_pfull_from_phalf(self.phalf)
        desired = self.pfull
        self.assertCoordsIdentical(actual, desired)

    def test_pfull_from_ps(self):
        for ps in [1e5, self.ps]:
            actual = self.coord_obj.pfull_from_ps(ps)
            desired = self.pfull*ps
            self.assertCoordsIdentical(actual, desired)

    def test_d_deta_from_phalf(self):
        actual = self.coord_obj.d_deta_from_phalf(self.phalf)
        desired = self.pfull
        self.assertCoordsIdentical(actual, desired)

    def test_d_deta_from_pfull(self):
        actual = self.coord_obj.d_deta_from_pfull(self.pfull)
        desired = self.pfull
        self.assertCoordsIdentical(actual, desired)

    def test_dp_from_ps(self):
        for ps in [1e5, self.ps]:
            actual = self.coord_obj.dp_from_ps(ps)
            desired = self.pfull*ps
            self.assertCoordsIdentical(actual, desired)

if __name__ == '__main__':
    sys.exit(unittest.main())
