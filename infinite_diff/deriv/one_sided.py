"""One-sided finite differencing for derivatives."""
import xarray as xr

from .. import OneSidedDiff, FwdDiff, BwdDiff
from . import FiniteDeriv


class OneSidedDeriv(FiniteDeriv):
    def __init__(self, arr, dim, coord=None):
        super(OneSidedDeriv, self).__init__(arr, dim, coord=coord)
        self._fin_diff_obj = OneSidedDiff(arr, dim)
        self._diff = self._fin_diff_obj.diff
        self._diff_rev = self._fin_diff_obj.diff_rev

    def _slice_edge(self, arr, spacing, order):
        raise NotImplementedError

    def _edge_deriv_rev(self, spacing, order):
        edge_arr = (
            self._diff_rev(arr=self._slice_edge(self.arr, spacing, order)) /
            self._diff_rev(arr=self._slice_edge(self.coord, spacing, order))
        )
        return self._slice_edge(edge_arr, spacing, order, pad=0)

    def _concat(self):
        raise NotImplementedError

    def deriv(self, spacing=1, order=1, fill_edge=False):
        """One-sided differencing approximation of derivative.

        :out: Array containing the df/dx approximation, with length in the 0th
            axis one less than that of the input array.
        """
        if order == 1:
            interior = self._deriv(spacing=spacing)
            if not fill_edge:
                return interior
            edge_arr = self._edge_deriv_rev(spacing, order)
            return self._concat(interior, edge_arr)
        elif order == 2:
            single_space = self.deriv(spacing=spacing, order=1,
                                      fill_edge=fill_edge)
            double_space = self.deriv(spacing=2*spacing, order=1,
                                      fill_edge=False)
            interior = 2*single_space - double_space
            if not fill_edge:
                return interior
            edge_arr = self._slice_edge(single_space, spacing, order, pad=0)
            return self._concat(interior, edge_arr)
        raise NotImplementedError("Forward differencing derivative only "
                                  "supported for 1st and 2nd order currently")


class FwdDeriv(OneSidedDeriv):
    def __init__(self, arr, dim, coord=None):
        super(FwdDeriv, self).__init__(arr, dim, coord=coord)
        self._fin_diff_obj = FwdDiff(arr, dim)
        self._diff = self._fin_diff_obj.diff
        self._diff_rev = self._fin_diff_obj.diff_rev

    def _slice_edge(self, arr, spacing, order, pad=1):
        return arr.isel(**{self.dim: slice(-(spacing*order + pad), None)})

    def _concat(self, interior, edge):
        return xr.concat([interior, edge], dim=self.dim)


class BwdDeriv(OneSidedDeriv):
    def __init__(self, arr, dim, coord=None):
        super(BwdDeriv, self).__init__(arr, dim, coord=coord)
        self._fin_diff_obj = BwdDiff(arr, dim)
        self._diff = self._fin_diff_obj.diff
        self._diff_rev = self._fin_diff_obj.diff_rev

    def _slice_edge(self, arr, spacing, order, pad=1):
        return arr.isel(**{self.dim: slice(None, (spacing*order + pad))})

    def _concat(self, interior, edge):
        return xr.concat([edge, interior], dim=self.dim)
