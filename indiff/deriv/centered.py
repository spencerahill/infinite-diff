from __future__ import division

import xarray as xr

from .. import CenDiff
from . import FiniteDeriv, FwdDeriv, BwdDeriv


class CenDeriv(FiniteDeriv):
    _DIFF_CLS = CenDiff
    _VALID_ORDERS = [2, 4]

    """Derivatives computed via centered finite differencing."""
    def __init__(self, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True):
        """
        :param arr: Data to be center-differenced.
        :type arr: `xarray.DataArray` or `xarray.Dataset`
        :param str dim: Dimension over which to compute.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given or None, arr[dim] is used.
        """
        super(CenDeriv, self).__init__(arr, dim, coord=coord, spacing=spacing,
                                       order=order, fill_edge=fill_edge)
        self._deriv_fwd_obj = FwdDeriv(arr, dim, coord=self.coord,
                                       spacing=self.spacing, order=2,
                                       fill_edge=self.fill_edge)
        self._deriv_bwd_obj = BwdDeriv(arr, dim, coord=self.coord,
                                       spacing=self.spacing, order=2,
                                       fill_edge=self.fill_edge)

    def _edge_deriv(self):
        left = self._deriv_bwd_obj._edge_deriv_rev()
        right = self._deriv_fwd_obj._edge_deriv_rev()
        pad = self.spacing*(self.order // 2)
        return (left[{self.dim: slice(None, pad)}],
                right[{self.dim: slice(-pad, None)}])

    def _concat(self, left, interior, right):
        return xr.concat([left, interior, right], dim=self.dim)

    def _deriv(self):
        """Lowest possible order derivative with this scheme."""
        interior = self._arr_diff_obj.diff() / self._coord_diff_obj.diff()
        if not self.fill_edge:
            return interior
        left, right = self._edge_deriv()
        return self._concat(left, interior, right)

    def deriv(self):
        """
        Centered differencing approximation of 1st derivative.

        :param int order: Order of accuracy to use.  Default 2.
        :param fill_edge: Whether or not to fill in the edge cells
            that don't have the needed neighbor cells for the stencil.  If
            `True`, use one-sided differencing with the same order of accuracy
            as `order`, and the outputted array is the same shape as `arr`.

            If `False`, the outputted array has a length in the computed axis
            reduced by `order`.
        """
        if self.order == 2:
            return self._deriv()
        if self.order == 4:
            single_space = self.__class__(self.arr, self.dim, coord=self.coord,
                                          spacing=self.spacing, order=2,
                                          fill_edge=self.fill_edge)._deriv()
            double_space = self.__class__(self.arr, self.dim, coord=self.coord,
                                          spacing=2*self.spacing, order=2,
                                          fill_edge=False)._deriv()
            interior = (4*single_space - double_space) / 3
            if not self.fill_edge:
                return interior
            left = single_space[{self.dim: slice(0, self.spacing*2)}]
            right = single_space[{self.dim: slice(-self.spacing*2, None)}]
            return self._concat(left, interior, right)
        raise NotImplementedError("Centered differencing only "
                                  "supported for 2nd and 4th order.")
