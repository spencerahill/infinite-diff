"""One-sided finite differencing for derivatives."""
import xarray as xr

from .. import OneSidedDiff, FwdDiff, BwdDiff
from . import FiniteDeriv


class OneSidedDeriv(FiniteDeriv):
    """Base class for one-sided differencing derivative approximations."""
    _DIFF_CLS = OneSidedDiff
    _DIFF_REV_CLS = OneSidedDiff
    _VALID_ORDERS = range(1, 3)

    def __init__(self, arr, dim, coord=None, spacing=1, order=1,
                 fill_edge=True):
        super(OneSidedDeriv, self).__init__(arr, dim, coord=coord,
                                            spacing=spacing, order=order,
                                            fill_edge=fill_edge)

    def _edge_deriv_rev(self):
        edge_arr = (self._DIFF_REV_CLS(self.arr, self.dim,
                                       spacing=self.spacing).diff() /
                    self._DIFF_REV_CLS(self.coord, self.dim,
                                       spacing=self.spacing).diff())
        return self._slice_edge(edge_arr)

    def _deriv(self):
        """Lowest possible order derivative with this scheme."""
        interior = self._arr_diff_obj.diff() / self._coord_diff_obj.diff()
        if not self.fill_edge:
            return interior
        edge_arr = self._edge_deriv_rev()
        return self._concat(interior, edge_arr)

    def deriv(self):
        """One-sided differencing approximation of derivative.

        :out: Array containing the derivative approximation
        """
        if self.order == 1:
            return self._deriv()
        if self.order == 2:
            single_space = self.__class__(self.arr, self.dim, coord=self.coord,
                                          spacing=self.spacing, order=1,
                                          fill_edge=self.fill_edge)._deriv()
            double_space = self.__class__(self.arr, self.dim, coord=self.coord,
                                          spacing=2*self.spacing, order=1,
                                          fill_edge=False)._deriv()
            interior = 2*single_space - double_space
            if not self.fill_edge:
                return interior
            edge_arr = self._slice_edge(single_space)
            return self._concat(interior, edge_arr)
        raise NotImplementedError("Forward differencing derivative only "
                                  "supported for 1st and 2nd order currently")


class FwdDeriv(OneSidedDeriv):
    """Derivatives using forward differencing."""
    _DIFF_CLS = FwdDiff
    _DIFF_REV_CLS = BwdDiff

    def __init__(self, arr, dim, coord=None, spacing=1, order=1,
                 fill_edge=True):
        super(FwdDeriv, self).__init__(arr, dim, coord=coord, spacing=spacing,
                                       order=order, fill_edge=fill_edge)

    def _slice_edge(self, arr):
        return arr[{self.dim: slice(-self.spacing*self.order, None)}]

    def _concat(self, interior, edge):
        return xr.concat([interior, edge], dim=self.dim)


class BwdDeriv(OneSidedDeriv):
    """Derivatives using backward differencing."""
    _DIFF_CLS = BwdDiff
    _DIFF_REV_CLS = FwdDiff

    def __init__(self, arr, dim, coord=None, spacing=1, order=1,
                 fill_edge=True):
        super(BwdDeriv, self).__init__(arr, dim, coord=coord, spacing=spacing,
                                       order=order, fill_edge=fill_edge)

    def _slice_edge(self, arr):
        return arr[{self.dim: slice(None, self.spacing*self.order)}]

    def _concat(self, interior, edge):
        return xr.concat([edge, interior], dim=self.dim)
