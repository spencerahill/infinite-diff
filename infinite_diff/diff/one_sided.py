"""Forward finite differencing."""
import xarray as xr

from ..utils import wraparound
from . import FiniteDiff


class OneSidedDiff(FiniteDiff):
    """One-sided finite differencing."""
    def __init__(self, arr, dim, spacing=1, wrap=False):
        super(OneSidedDiff, self).__init__(arr, dim, spacing=spacing)

    def _wrap(self):
        raise NotImplementedError

    def diff(self):
        """One-sided differencing."""
        left = self._slice_arr_dim(slice(0, -self.spacing), self.arr)
        right = self._slice_arr_dim(slice(self.spacing, None), self.arr)
        return xr.DataArray(right.values, dims=left.dims,
                            coords=left.coords) - left


class FwdDiff(OneSidedDiff):
    """Forward finite differencing."""
    def __init__(self, arr, dim, spacing=1, wrap=False):
        super(FwdDiff, self).__init__(arr, dim, spacing=spacing)

    def _wrap(self):
        return wraparound(self.arr, self.dim, left_to_right=self.spacing,
                          right_to_left=0, circumf=0, spacing=1)


class BwdDiff(OneSidedDiff):
    """Backward finite differencing."""
    def __init__(self, arr, dim, spacing=1, wrap=False):
        super(BwdDiff, self).__init__(arr, dim, spacing=spacing)

    def _wrap(self):
        return wraparound(self.arr, self.dim, left_to_right=0,
                          right_to_left=self.spacing, circumf=0, spacing=1)

    def diff(self):
        """One sided differencing in the opposite direction."""
        arr = self._reverse_dim(self.arr)
        return -1*self._reverse_dim(
            FwdDiff(arr, self.dim, spacing=self.spacing).diff()
        )
