"""Centered finite differencing."""
import xarray as xr

from . import FiniteDiff, BwdDiff, FwdDiff


class CenDiff(FiniteDiff):
    """Centered finite differencing."""
    _MIN_SPACING_FACTOR = 2
    _DIFF_BWD_CLS = BwdDiff
    _DIFF_FWD_CLS = FwdDiff

    def __init__(self, arr, dim, spacing=1, fill_edge=False):
        super(CenDiff, self).__init__(arr, dim, spacing=spacing)
        self.fill_edge = fill_edge
        self._diff_bwd = self._DIFF_BWD_CLS(arr, dim, spacing=spacing).diff
        self._diff_fwd = self._DIFF_FWD_CLS(arr, dim, spacing=spacing).diff

    def _diff_edge(self, side='left'):
        """One-sided differencing of array edge."""
        if side == 'left':
            trunc = slice(0, self.spacing + 1)
            cls = self._DIFF_FWD_CLS
        elif side == 'right':
            trunc = slice(-(self.spacing + 1), None)
            cls = self._DIFF_BWD_CLS
        else:
            raise ValueError("Parameter `side` must be either 'left' "
                             "or 'right': {}").format(side)
        arr_edge = self._slice_arr_dim(trunc, self.arr)
        return cls(arr_edge, self.dim, spacing=self.spacing).diff()

    def diff(self):
        """Centered differencing of the DataArray or Dataset.

        :param fill_edge: Whether or not to fill in the edge cells
            that don't have the needed neighbor cells for the stencil.  If
            `True`, use one-sided differencing with the same order of accuracy
            as `order`, and the outputted array is the same shape as `arr`.

            If `'left'` or `'right'`, fill only that side.

            If `False`, the outputted array has a length in the computed axis
            reduced by `order`.
        """
        left = self._slice_arr_dim(slice(0, -self.spacing), self.arr)
        right = self._slice_arr_dim(slice(self.spacing, None), self.arr)
        interior = (self._DIFF_FWD_CLS(right, self.dim, self.spacing).diff() +
                    self._DIFF_BWD_CLS(left, self.dim, self.spacing).diff())

        if self.fill_edge in ('left', 'both'):
            diff_left = self._diff_edge(side='left')
            interior = xr.concat([diff_left, interior], dim=self.dim)
        if self.fill_edge == ('right', 'both'):
            diff_right = self._diff_edge(side='right')
            interior = xr.concat([interior, diff_right], dim=self.dim)
        return interior
