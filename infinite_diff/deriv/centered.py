from __future__ import division

import xarray as xr

from .. import CenDiff
from . import FiniteDeriv, FwdDeriv, BwdDeriv


class CenDeriv(FiniteDeriv):
    """Derivatives computed via centered finite differencing."""
    def __init__(self, arr, dim, coord=None):
        """
        :param arr: Data to be center-differenced.
        :type arr: `xarray.DataArray` or `xarray.Dataset`
        :param str dim: Dimension over which to compute.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given or None, arr[dim] is used.
        """
        super(CenDeriv, self).__init__(arr, dim, coord=coord)
        self._fin_diff_obj = CenDiff(arr, dim)
        self._diff = self._fin_diff_obj.diff

        self._deriv_fwd_obj = FwdDeriv(arr, dim, coord=coord)
        self._deriv_bwd_obj = BwdDeriv(arr, dim, coord=coord)

    def _edge_deriv(self, spacing, order):
        left = self._deriv_bwd_obj._edge_deriv_rev(spacing, order)
        right = self._deriv_fwd_obj._edge_deriv_rev(spacing, order)
        pad = spacing*(order // 2)
        return (left.isel(**{self.dim: slice(None, pad)}),
                right.isel(**{self.dim: slice(-pad, None)}))

    def _concat(self, left, interior, right):
        return xr.concat([left, interior, right], dim=self.dim)

    def deriv(self, spacing=1, order=2, fill_edge=False):
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
        if order == 2:
            interior = self._deriv(spacing=spacing)
            if not fill_edge:
                return interior
            left, right = self._edge_deriv(spacing, 2)
            return self._concat(left, interior, right)
        elif order == 4:
            single_space = self.deriv(spacing=spacing, order=2,
                                      fill_edge=fill_edge)
            double_space = self.deriv(spacing=2*spacing, order=2,
                                      fill_edge=False)
            interior = (4*single_space - double_space) / 3
            if not fill_edge:
                return interior
            left = self._deriv_bwd_obj._slice_edge(
                single_space, spacing, 2, pad=0
            )
            right = self._deriv_fwd_obj._slice_edge(
                single_space, spacing, 2, pad=0
            )
            return self._concat(left, interior, right)
        raise NotImplementedError("Centered differencing only "
                                  "supported for 2nd and 4th order.")
