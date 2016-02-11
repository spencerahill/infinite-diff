import sys
import unittest

from infinite_diff.coord import Coord, HorizCoord, XCoord, YCoord, Lon, Lat
from infinite_diff.coord import VertCoord, ZCoord, Pressure, Sigma, Eta

from . import InfiniteDiffTestCase


class CoordTestCase(InfiniteDiffTestCase):
    def _make_coord_obj(self):
        return self.coord_cls(self.arr, dim=self.dim, cyclic=self.cyclic)

    def setUp(self):
        super(CoordTestCase, self).setUp()
        self.arr = self.arange
        self.cyclic = False
        self.coord_cls = Coord


class TestCoord(CoordTestCase):
    def setUp(self):
        super(TestCoord, self).setUp()
        self.coord_obj = self._make_coord_obj()

    def test_init(self):
        self.assertIsInstance(self.coord_obj, self.coord_cls)
        self.assertDatasetIdentical(self.coord_obj._arr, self.arr)

    def test_getitem(self):
        for key in range(self.array_len):
            self.assertDatasetIdentical(self.coord_obj[:, key],
                                        self.coord_obj._arr[:, key])

    def test_deriv_prefactor(self):
        self.assertNotImplemented(self.coord_obj.deriv_prefactor)

    def test_deriv_factor(self):
        self.assertNotImplemented(self.coord_obj.deriv_factor)


class HorizCoordTestCase(CoordTestCase):
    def setUp(self):
        super(HorizCoordTestCase, self).setUp()
        self.coord_cls = HorizCoord


class TestHorizCoord(HorizCoordTestCase, TestCoord):
    def setUp(self):
        super(TestHorizCoord, self).setUp()
        self.coord_obj = self._make_coord_obj()


class XCoordTestCase(HorizCoordTestCase):
    def setUp(self):
        super(XCoordTestCase, self).setUp()
        self.coord_cls = XCoord


class TestXCoord(XCoordTestCase, TestHorizCoord):
    def setUp(self):
        super(TestXCoord, self).setUp()
        self.coord_obj = self._make_coord_obj()

    def test_deriv_prefactor(self):
        self.assertEqual(self.coord_obj.deriv_prefactor(), 1.)

    def test_deriv_factor(self):
        self.assertEqual(self.coord_obj.deriv_factor(), 1.)


class YCoordTestCase(XCoordTestCase):
    def setUp(self):
        super(YCoordTestCase, self).setUp()
        self.coord_cls = YCoord


class TestYCoord(YCoordTestCase, TestXCoord):
    def setUp(self):
        super(TestXCoord, self).setUp()
        self.coord_obj = self._make_coord_obj()


class LonTestCase(XCoordTestCase):
    def setUp(self):
        super(LonTestCase, self).setUp()
        self.cyclic = True
        self.coord_cls = Lon


class TestLon(LonTestCase, TestXCoord):
    def setUp(self):
        super(TestLon, self).setUp()
        self.coord_obj = self._make_coord_obj()

    @unittest.skip("Not implemented yet")
    def test_deriv_prefactor(self):
        raise NotImplementedError


class LatTestCase(LonTestCase):
    def setUp(self):
        super(LatTestCase, self).setUp()
        self.cyclic = True
        self.coord_cls = Lat


class TestLat(LatTestCase, TestLon):
    def setUp(self):
        super(TestLat, self).setUp()
        self.coord_obj = self._make_coord_obj()

    @unittest.skip("Not implemented yet")
    def test_deriv_prefactor(self):
        raise NotImplementedError

    @unittest.skip("Not implemented yet")
    def test_deriv_factor(self):
        raise NotImplementedError


class VertCoordTestCase(CoordTestCase):
    def _make_coord_obj(self):
        return self.coord_cls(self.arr, dim=self.dim)

    def setUp(self):
        super(VertCoordTestCase, self).setUp()
        self.coord_cls = VertCoord


class TestVertCoord(VertCoordTestCase, TestCoord):
    def setUp(self):
        super(TestVertCoord, self).setUp()
        self.coord_obj = self._make_coord_obj()


class ZCoordTestCase(VertCoordTestCase):
    def setUp(self):
        super(ZCoordTestCase, self).setUp()
        self.coord_cls = ZCoord


class TestZCoord(ZCoordTestCase, TestCoord):
    def setUp(self):
        super(TestZCoord, self).setUp()
        self.coord_obj = self._make_coord_obj()


class PressureTestCase(VertCoordTestCase):
    def setUp(self):
        super(PressureTestCase, self).setUp()
        self.coord_cls = Pressure


class TestPressure(PressureTestCase, TestCoord):
    def setUp(self):
        super(TestPressure, self).setUp()
        self.coord_obj = self._make_coord_obj()


class SigmaTestCase(VertCoordTestCase):
    def setUp(self):
        super(SigmaTestCase, self).setUp()
        self.coord_cls = Sigma


class TestSigma(SigmaTestCase, TestCoord):
    def setUp(self):
        super(TestSigma, self).setUp()
        self.coord_obj = self._make_coord_obj()


class EtaTestCase(VertCoordTestCase):
    def _make_coord_obj(self, pk, bk):
        return self.coord_cls(self.arr, pk, bk, dim=self.dim)

    def setUp(self):
        super(EtaTestCase, self).setUp()
        self.pk = []
        self.bk = []
        self.coord_cls = Eta


class TestEta(EtaTestCase, TestCoord):
    def setUp(self):
        super(TestEta, self).setUp()
        self.coord_obj = self._make_coord_obj(self.pk, self.bk)

if __name__ == '__main__':
    sys.exit(unittest.main())
