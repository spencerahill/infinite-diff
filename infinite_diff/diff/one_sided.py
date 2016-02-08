"""Forward finite differencing."""
import xarray as xr

from . import FiniteDiff


class OneSidedDiff(FiniteDiff):
    """One-sided finite differencing."""
    def __init__(self, arr, dim):
        super(OneSidedDiff, self).__init__(arr, dim)

    def _diff(self, arr=None, spacing=1):
        """One-sided differencing."""
        arr = self._find_arr(arr)
        self._check_spacing(spacing)
        self._check_arr_len(arr=arr, spacing=spacing)
        left = self._slice_arr_dim(slice(0, -spacing), arr=arr)
        right = self._slice_arr_dim(slice(spacing, None), arr=arr)
        return xr.DataArray(right.values, dims=left.dims,
                            coords=left.coords) - left

    def _diff_rev(self, arr=None, spacing=1):
        """One sided differencing in the opposite direction."""
        arr = self._find_arr(arr)
        arr = self._reverse_dim(arr=arr)
        return -1*self._reverse_dim(arr=self._diff(arr=arr, spacing=spacing))


class FwdDiff(OneSidedDiff):
    """Forward finite differencing."""
    def __init__(self, arr, dim):
        super(FwdDiff, self).__init__(arr, dim)
        self.diff = self._diff
        self.diff_rev = self._diff_rev


class BwdDiff(OneSidedDiff):
    """Backward finite differencing."""
    def __init__(self, arr, dim):
        super(BwdDiff, self).__init__(arr, dim)
        self.diff = self._diff_rev
        self.diff_rev = self._diff
