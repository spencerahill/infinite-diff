"""One-sided finite differencing for derivatives."""
import xarray as xr

from .. import OneSidedDiff, FwdDiff, BwdDiff
from . import FiniteDeriv


class OneSidedDeriv(FiniteDeriv):
    """Base class for one-sided differencing derivative approximations."""
    _DIFF_CLS = OneSidedDiff
    _DIFF_REV_CLS = OneSidedDiff

    def __init__(self, arr, dim, coord=None):
        super(OneSidedDeriv, self).__init__(arr, dim, coord=coord)
        self._arr_diff_rev = self._arr_diff_obj.diff_rev
        self._coord_diff_rev = self._coord_diff_obj.diff_rev

    def _edge_deriv_rev(self, spacing, order):
        edge_arr = (
            self._DIFF_REV_CLS(self._slice_edge(self.arr, spacing, order),
                               self.dim).diff(spacing=spacing) /
            self._DIFF_REV_CLS(self._slice_edge(self.coord, spacing, order),
                               self.dim).diff(spacing=spacing)
        )
        return self._slice_edge(edge_arr, spacing, order, pad=0)

    def deriv(self, spacing=1, order=1, fill_edge=False):
        """One-sided differencing approximation of derivative.

        :out: Array containing the derivative approximation
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
    """Derivatives using forward differencing."""
    _DIFF_CLS = FwdDiff
    _DIFF_REV_CLS = BwdDiff

    def __init__(self, arr, dim, coord=None):
        super(FwdDeriv, self).__init__(arr, dim, coord=coord)

    def _slice_edge(self, arr, spacing, order, pad=1):
        return arr.isel(**{self.dim: slice(-(spacing*order + pad), None)})

    def _concat(self, interior, edge):
        return xr.concat([interior, edge], dim=self.dim)


class BwdDeriv(OneSidedDeriv):
    """Derivatives using backward differencing."""
    _DIFF_CLS = BwdDiff
    _DIFF_REV_CLS = FwdDiff

    def __init__(self, arr, dim, coord=None):
        super(BwdDeriv, self).__init__(arr, dim, coord=coord)

    def _slice_edge(self, arr, spacing, order, pad=1):
        return arr.isel(**{self.dim: slice(None, (spacing*order + pad))})

    def _concat(self, interior, edge):
        return xr.concat([edge, interior], dim=self.dim)
