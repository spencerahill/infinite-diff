import sys
import unittest

from infinite_diff.advec import PhysUpwind, LonUpwind
from infinite_diff.deriv import PhysDeriv, LonFwdDeriv, LonBwdDeriv

from . import InfiniteDiffTestCase


class PhysAdvecSharedTests(object):
    def test_init(self):
        self.assertIsInstance(self.advec_obj._deriv_bwd_obj,
                              self._DERIV_BWD_CLS)
        self.assertIsInstance(self.advec_obj._deriv_fwd_obj,
                              self._DERIV_FWD_CLS)

    def test_advec(self):
        self.assertNotImplemented(self.method)


class PhysUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = PhysUpwind
    _DERIV_FWD_CLS = PhysDeriv
    _DERIV_BWD_CLS = PhysDeriv

    def setUp(self):
        super(PhysUpwindTestCase, self).setUp()
        self.arr = self.random
        self.flow = self.random2
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self.dim)
        self.method = self.advec_obj.advec


class TestPhysUpwind(PhysAdvecSharedTests, PhysUpwindTestCase):
    pass


class LonUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = LonUpwind
    _DERIV_FWD_CLS = LonFwdDeriv
    _DERIV_BWD_CLS = LonBwdDeriv

    def setUp(self):
        super(LonUpwindTestCase, self).setUp()
        self.arr = self.random
        self.flow = self.random2
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self.dim)
        self.method = self.advec_obj.advec


class TestLonUpwind(PhysAdvecSharedTests, LonUpwindTestCase):
    def test_advec(self):
        self.method()


if __name__ == '__main__':
    sys.exit(unittest.main())
