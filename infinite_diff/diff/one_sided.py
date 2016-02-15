"""Forward finite differencing."""
import xarray as xr

from . import FiniteDiff


class OneSidedDiff(FiniteDiff):
    """One-sided finite differencing."""
    def __init__(self, arr, dim):
        super(OneSidedDiff, self).__init__(arr, dim)

    def _diff(self, spacing=1):
        """One-sided differencing."""
        self._check_spacing(spacing)
        self._check_arr_len(spacing=spacing)
        left = self._slice_arr_dim(slice(0, -spacing))
        right = self._slice_arr_dim(slice(spacing, None))
        return xr.DataArray(right.values, dims=left.dims,
                            coords=left.coords) - left

    def _diff_rev(self, spacing=1):
        """One sided differencing in the opposite direction."""
        arr = self._reverse_dim(self.arr)
        return -1*self._reverse_dim(
            self.__class__(arr, self.dim)._diff(spacing=spacing)
        )

    def diff(self, arr=None, spacing=1):
        raise NotImplementedError

    def diff_rev(self, arr=None, spacing=1):
        raise NotImplementedError


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
