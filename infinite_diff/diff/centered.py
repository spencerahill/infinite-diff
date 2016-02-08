"""Centered finite differencing."""
import xarray as xr

from . import FiniteDiff, BwdDiff, FwdDiff


class CenDiff(FiniteDiff):
    """Centered finite differencing."""
    def __init__(self, arr, dim):
        super(CenDiff, self).__init__(arr, dim)
        self._bwd_diff = BwdDiff(arr, dim).diff
        self._fwd_diff = FwdDiff(arr, dim).diff

    def _diff_edge(self, spacing=1, side='left'):
        """One-sided differencing of array edge."""
        if side == 'left':
            trunc = slice(0, spacing+1)
            method = self._fwd_diff
        elif side == 'right':
            trunc = slice(-(spacing+1), None)
            method = self._bwd_diff
        else:
            raise ValueError("Parameter `side` must be either 'left' "
                             "or 'right': {}").format(side)
        arr_edge = self._slice_arr_dim(trunc)
        return method(arr=arr_edge)

    def diff(self, arr=None, spacing=1, fill_edge=False):
        """Centered differencing of the DataArray or Dataset.

        :param fill_edge: Whether or not to fill in the edge cells
            that don't have the needed neighbor cells for the stencil.  If
            `True`, use one-sided differencing with the same order of accuracy
            as `order`, and the outputted array is the same shape as `arr`.

            If `'left'` or `'right'`, fill only that side.

            If `False`, the outputted array has a length in the computed axis
            reduced by `order`.
        """
        self._check_spacing()
        self._check_arr_len(2*spacing)

        left = self._slice_arr_dim(slice(0, -spacing))
        right = self._slice_arr_dim(slice(spacing, None))
        diff = self._fwd_diff(arr=right) + self._bwd_diff(arr=left)

        if fill_edge in ('left', 'both'):
            diff_left = self._diff_edge(side='left')
            diff = xr.concat([diff_left, diff], dim=self.dim)
        if fill_edge == ('right', 'both'):
            diff_right = self._diff_edge(side='right')
            diff = xr.concat([diff, diff_right], dim=self.dim)
        return diff
