"""Finite differencing."""


class FiniteDiff(object):
    """Base class for finite differencing of xarray objects."""
    def __init__(self, arr, dim):
        """
        Create a `FiniteDiff` object.

        :param arr: Data to be center-differenced.
        :type arr: `xarray.DataArray` or `xarray.Dataset`
        :param str dim: Dimension over which to perform the differencing.
        :param int spacing: How many gridpoints over to use.  Size of resulting
            array depends on this value.
        """
        self.arr = arr
        self.dim = dim

    @staticmethod
    def _check_spacing(spacing, min_spacing=1):
        """Ensure spacing value is valid."""
        msg = ("'spacing' value of {} invalid; spacing must be positive "
               "integer".format(spacing))
        if not isinstance(spacing, int):
            raise TypeError(msg)
        if spacing < min_spacing:
            raise ValueError(msg)

    def _find_arr(self, arr):
        if arr is None:
            return self.arr
        return arr

    def _check_arr_len(self, arr=None, spacing=1, pad=1):
        """Ensure array is long enough to perform the differencing."""
        arr = self._find_arr(arr)
        try:
            len_arr_dim = len(arr[self.dim])
        except TypeError:
            len_arr_dim = 0
        if len_arr_dim < spacing + pad:
            msg = ("Array along dim '{}' is too small (={}) for differencing "
                   "with spacing {}".format(self.dim, len_arr_dim, spacing))
            raise ValueError(msg)

    def _slice_arr_dim(self, slice_, arr=None):
        """Get a slice of a DataArray along a particular dim."""
        arr = self._find_arr(arr)
        return arr.isel(**{self.dim: slice_})

    def _reverse_dim(self, arr=None):
        """Reverse the DataArray along the given dimension."""
        arr = self._find_arr(arr)
        return self._slice_arr_dim(slice(-1, None, -1), arr=arr)

    def _diff(self, arr=None, spacing=1):
        raise NotImplementedError

    def _diff_rev(self, arr=None, spacing=1):
        raise NotImplementedError

    def diff(self, arr=None, spacing=1):
        raise NotImplementedError

    def diff_rev(self, arr=None, spacing=1):
        raise NotImplementedError
