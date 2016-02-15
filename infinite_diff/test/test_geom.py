import sys
import unittest

from infinite_diff.coord import VertCoord, Eta
from infinite_diff.geom import HorizGeom, HorizCartesian, HorizSphere
from infinite_diff.geom import HorizVertGeom, SphereEtaGeom

from . import InfiniteDiffTestCase


class HorizGeomSharedTests(object):
    def test_init(self):
        self.assertIsInstance(self.geom_obj.x, self._GEOM_CLS._X_COORD_CLS)
        self.assertIsInstance(self.geom_obj.y, self._GEOM_CLS._Y_COORD_CLS)


class HorizGeomTestCase(InfiniteDiffTestCase):
    _GEOM_CLS = HorizGeom
    _X_COORD_CLS = _GEOM_CLS._X_COORD_CLS
    _Y_COORD_CLS = _GEOM_CLS._Y_COORD_CLS

    def setUp(self):
        super(HorizGeomTestCase, self).setUp()
        self.x = self._X_COORD_CLS(self.arange, self.dim)
        self.y = self._Y_COORD_CLS(self.arange, self.dim)
        self.geom_obj = HorizGeom(self.x, self.y)


class TestHorizGeom(HorizGeomTestCase, HorizGeomSharedTests):
    pass


class HorizCartesianTestCase(HorizGeomTestCase):
    _GEOM_CLS = HorizCartesian
    _X_COORD_CLS = _GEOM_CLS._X_COORD_CLS
    _Y_COORD_CLS = _GEOM_CLS._Y_COORD_CLS

    def setUp(self):
        super(HorizCartesianTestCase, self).setUp()
        self.geom_obj = HorizCartesian(self.x, self.y)


class TestHorizCartesian(HorizCartesianTestCase, TestHorizGeom):
    pass


class HorizSphereTestCase(HorizGeomTestCase):
    _GEOM_CLS = HorizSphere
    _X_COORD_CLS = _GEOM_CLS._X_COORD_CLS
    _Y_COORD_CLS = _GEOM_CLS._Y_COORD_CLS

    def setUp(self):
        super(HorizSphereTestCase, self).setUp()
        self.geom_obj = HorizSphere(self.x, self.y)


class TestHorizSphere(HorizSphereTestCase, TestHorizGeom):
    pass


# class HorizVertGeomTestCase(InfiniteDiffTestCase):
#     _GEOM_CLS = HorizVertGeom

#     def setUp(self):
#         super(HorizSphereTestCase, self).setUp()
#         self.geom_obj = HorizVertGeomTestCase(self.x, self.y, self.z)


# class TestHorizVertGeomTestCase(HorizVertGeomTestCase):
#     pass


if __name__ == '__main__':
    sys.exit(unittest.main())
