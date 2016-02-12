import sys
import unittest

from infinite_diff.coord import Coord, HorizCoord, XCoord, YCoord, Lon, Lat
from infinite_diff.coord import VertCoord, ZCoord, Pressure, Sigma, Eta

from . import InfiniteDiffTestCase


class CoordSharedTests(object):
    def test_init(self):
        self.assertIsInstance(self.coord_obj, self._COORD_CLS)
        self.assertDatasetIdentical(self.coord_obj._arr, self.arr)

    def test_getitem(self):
        for key in range(self.array_len):
            self.assertDatasetIdentical(self.coord_obj[:, key],
                                        self.coord_obj._arr[:, key])

    def test_deriv_prefactor(self):
        self.assertNotImplemented(self.coord_obj.deriv_prefactor)

    def test_deriv_factor(self):
        self.assertNotImplemented(self.coord_obj.deriv_factor)


class CoordTestCase(InfiniteDiffTestCase):
    _COORD_CLS = Coord
    _CYCLIC = False

    def setUp(self):
        super(CoordTestCase, self).setUp()
        self.arr = self.arange
        self.coord_obj = Coord(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class TestCoord(CoordTestCase, CoordSharedTests):
    pass


class HorizCoordTestCase(CoordTestCase):
    _COORD_CLS = HorizCoord

    def setUp(self):
        super(HorizCoordTestCase, self).setUp()
        self.coord_obj = HorizCoord(self.arr, dim=self.dim,
                                    cyclic=self._CYCLIC)


class TestHorizCoord(HorizCoordTestCase, TestCoord):
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


class TestYCoord(YCoordTestCase, TestXCoord):
    def setUp(self):
        super(TestYCoord, self).setUp()
        self.coord_obj = YCoord(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class LonTestCase(XCoordTestCase):
    _COORD_CLS = Lon
    _CYCLIC = True

    def setUp(self):
        super(LonTestCase, self).setUp()
        self.coord_obj = Lon(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class TestLon(LonTestCase, TestXCoord):
    @unittest.skip("Not implemented yet")
    def test_deriv_prefactor(self):
        raise NotImplementedError


class LatTestCase(YCoordTestCase):
    _COORD_CLS = Lat

    def setUp(self):
        super(LatTestCase, self).setUp()
        self.coord_obj = Lat(self.arr, dim=self.dim, cyclic=self._CYCLIC)


class TestLat(LatTestCase, TestYCoord):
    @unittest.skip("Not implemented yet")
    def test_deriv_prefactor(self):
        raise NotImplementedError

    @unittest.skip("Not implemented yet")
    def test_deriv_factor(self):
        raise NotImplementedError


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
        self.coord_obj = Pressure(self.arr, dim=self.dim)


class TestPressure(PressureTestCase, TestVertCoord):
    pass


class SigmaTestCase(VertCoordTestCase):
    _COORD_CLS = Sigma

    def setUp(self):
        super(SigmaTestCase, self).setUp()
        self.coord_obj = Sigma(self.arr, self.arr, dim=self.dim)


class TestSigma(SigmaTestCase, TestVertCoord):
    pass


class EtaTestCase(VertCoordTestCase):
    _COORD_CLS = Eta

    def setUp(self):
        super(EtaTestCase, self).setUp()
        self.pk = []
        self.bk = []
        self.coord_obj = Eta(self.arr, self.pk, self.bk, dim=self.dim)


class TestEta(EtaTestCase, TestVertCoord):
    pass


if __name__ == '__main__':
    sys.exit(unittest.main())
