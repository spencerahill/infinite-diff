"""Testing module of infinite_diff."""
import unittest

import numpy as np
import xarray as xr


class InfiniteDiffTestCase(unittest.TestCase):
    def setUp(self):
        self.array_len = 10
        self.dim = 'testdim'
        self.dummy_len = 3
        self.dummy_dim = 'dummydim'

        self.ones = xr.DataArray(np.ones((self.dummy_len, self.array_len)),
                                 dims=[self.dummy_dim, self.dim])

        self.zeros = xr.DataArray(np.zeros(self.ones.shape),
                                  dims=self.ones.dims)
        self.arange = xr.DataArray(
            np.arange(self.array_len*self.dummy_len).reshape(self.ones.shape),
            dims=self.ones.dims
        )
        randstate = np.random.RandomState(12345)
        self.random = xr.DataArray(randstate.rand(*self.ones.shape),
                                   dims=self.ones.dims,
                                   coords=self.ones.coords)
        randstate = np.random.RandomState(54321)
        self.random2 = xr.DataArray(randstate.rand(*self.ones.shape),
                                    dims=self.ones.dims,
                                    coords=self.ones.coords)

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(None, -(n+1))})
                            for n in range(self.array_len)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(None, -(n+1))})
                           for n in range(self.array_len)]
        self.arange_trunc = [self.arange.isel(**{self.dim:
                                                 slice(None, -(n+1))})
                             for n in range(self.array_len)]
        self.random_trunc = [self.random.isel(**{self.dim:
                                                 slice(None, -(n+1))})
                             for n in range(self.array_len)]

    def tearDown(self):
        pass

    def assertDatasetIdentical(self, d1, d2):
        assert d1.identical(d2), (d1, d2)

    def assertNotImplemented(self, func, *args, **kwargs):
        self.assertRaises(NotImplementedError, func, *args, **kwargs)
