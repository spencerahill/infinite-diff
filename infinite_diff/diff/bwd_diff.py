"""Backward finite differencing."""
import xarray as xr

from . import FiniteDiff


class BwdDiff(FiniteDiff):
    """Backward finite differencing."""
    def __init__(self, arr, dim):
        super(BwdDiff, self).__init__(arr, dim)

    def diff(self, arr=None, spacing=1):
        """Backward differencing of the array."""
        if arr is None:
            arr = self.arr
        self._check_spacing(spacing)
        self._check_arr_len(arr=arr, spacing=spacing)
        left = self._slice_arr_dim(slice(0, -spacing))
        right = self._slice_arr_dim(slice(spacing, None))
        return right - xr.DataArray(left.values, dims=right.dims,
                                    coords=right.coords)
