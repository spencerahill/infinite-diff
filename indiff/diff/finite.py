"""Finite differencing."""


class FiniteDiff(object):
    """Base class for finite differencing of xarray objects."""
    def __init__(self, arr, dim, spacing=1, wrap=False):
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
        self.spacing = spacing
        self.wrap = wrap

    def _slice_arr_dim(self, slice_, arr):
        """Get a slice of a DataArray along a particular dim."""
        return arr[{self.dim: slice_}]

    def _reverse_dim(self, arr):
        """Reverse the DataArray along the given dimension."""
        return self._slice_arr_dim(slice(-1, None, -1), arr)

    def _wrap(self):
        raise NotImplementedError

    def diff(self):
        raise NotImplementedError
