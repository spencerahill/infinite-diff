#! /usr/bin/env python
"""Tests"""
import sys
import unittest

import numpy as np

from finite_diff import FiniteDiff


class FiniteDiffTestCase(unittest.TestCase):
    def setUp(self):
        self._array_len = 10
        self.ones = np.ones(self._array_len)
        self.ones_trunc1 = self.ones[1:]
        self.ones_trunc2 = self.ones[2:]
        self.ones_trunc3 = self.ones[3:]
        self.ones_trunc4 = self.ones[4:]
        self.zeros = np.zeros(self._array_len)
        self.zeros_trunc1 = self.zeros[:-1]
        self.zeros_trunc2 = self.zeros[:-2]
        self.zeros_trunc3 = self.zeros[:-3]
        self.zeros_trunc4 = self.zeros[:-4]
        self.arange = np.arange(self._array_len)
        # self.arange_trunc_left = self.arange[1:]
        # self.arange_trunc_right = self.arange[:-1]

    def tearDown(self):
        pass


class TestFiniteDiff(FiniteDiffTestCase):
    def test_fwd_diff1_bad_input(self):
        self.assertRaises(ValueError, FiniteDiff.fwd_diff1,
                          self.ones, self.ones)
        self.assertRaises(ValueError, FiniteDiff.fwd_diff1,
                          self.ones, self.zeros)
        self.assertRaises(ValueError, FiniteDiff.fwd_diff1, self.ones, 0.)

    def test_fwd_diff1_zero_slope(self):
        np.testing.assert_array_equal(FiniteDiff.fwd_diff1(self.ones,
                                                           self.arange),
                                      self.zeros_trunc1)
        np.testing.assert_array_equal(FiniteDiff.fwd_diff1(self.zeros,
                                                           self.arange),
                                      self.zeros_trunc1)
        np.testing.assert_array_equal(FiniteDiff.fwd_diff1(self.zeros, 1.),
                                      self.zeros_trunc1)

    def test_fwd_diff1_constant_slope(self):
        np.testing.assert_array_equal(FiniteDiff.fwd_diff1(self.arange, 1.),
                                      self.ones_trunc1)
        np.testing.assert_array_equal(FiniteDiff.fwd_diff1(self.arange*-2.5,
                                                           1.),
                                      self.ones_trunc1*-2.5)

    def test_fwd_diff2_zero_slope(self):
        np.testing.assert_array_equal(FiniteDiff.fwd_diff2(self.ones,
                                                           self.arange),
                                      self.zeros_trunc2)
        np.testing.assert_array_equal(FiniteDiff.fwd_diff2(self.zeros,
                                                           self.arange),
                                      self.zeros_trunc2)
        np.testing.assert_array_equal(FiniteDiff.fwd_diff2(self.zeros, 1.),
                                      self.zeros_trunc2)

    def test_fwd_diff2_constant_slope(self):
        np.testing.assert_array_equal(FiniteDiff.fwd_diff2(self.arange, 1.),
                                      self.ones_trunc2)
        np.testing.assert_array_equal(FiniteDiff.fwd_diff2(self.arange*-2.5, 1.),
                                      self.ones_trunc2*-2.5)

    def test_cen_diff2_zero_slope(self):
        np.testing.assert_array_equal(FiniteDiff.cen_diff2(self.ones,
                                                           self.arange),
                                      self.zeros_trunc2)
        np.testing.assert_array_equal(FiniteDiff.cen_diff2(self.zeros,
                                                           self.arange),
                                      self.zeros_trunc2)
        np.testing.assert_array_equal(FiniteDiff.cen_diff2(self.zeros, 1.),
                                      self.zeros_trunc2)

    def test_cen_diff2_constant_slope(self):
        np.testing.assert_array_equal(FiniteDiff.cen_diff2(self.arange, 1.),
                                      self.ones_trunc2)
        np.testing.assert_array_equal(FiniteDiff.cen_diff2(self.arange*-2.5,
                                                           1.),
                                      self.ones_trunc2*-2.5)

    def test_cen_diff4_zero_slope(self):
        np.testing.assert_array_equal(FiniteDiff.cen_diff4(self.ones,
                                                           self.arange),
                                      self.zeros_trunc4)
        np.testing.assert_array_equal(FiniteDiff.cen_diff4(self.zeros,
                                                           self.arange),
                                      self.zeros_trunc4)
        np.testing.assert_array_equal(FiniteDiff.cen_diff4(self.zeros, 1.),
                                      self.zeros_trunc4)

    def test_cen_diff4_constant_slope(self):
        np.testing.assert_array_equal(FiniteDiff.cen_diff4(self.arange, 1.),
                                      self.ones_trunc4)
        np.testing.assert_array_equal(FiniteDiff.cen_diff4(self.arange*-2.5,
                                                           1.),
                                      self.ones_trunc4*-2.5)


if __name__ == '__main__':
    sys.exit(unittest.main())
