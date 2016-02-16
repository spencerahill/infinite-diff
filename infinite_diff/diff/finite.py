"""Finite differencing."""


class FiniteDiff(object):
    """Base class for finite differencing of xarray objects."""
    _MIN_SPACING_FACTOR = 1

    # def _check_arr_len(self):
    #     """Ensure array is long enough to perform the differencing."""
    #     try:
    #         len_arr_dim = len(self.arr[self.dim])
    #     except TypeError:
    #         len_arr_dim = 0
    #     print(self.arr[self.dim])
    #     msg = ("Array along dim '{}' is too small (={}) for differencing "
    #            "with spacing {}".format(self.dim, len_arr_dim, self.spacing))
    #     if len_arr_dim < self.spacing*self._MIN_SPACING_FACTOR + 1:
    #         raise ValueError(msg)

    # def _check_spacing(self):
    #     """Ensure spacing value is valid."""
    #     msg = ("'spacing' value of {} invalid; spacing must be positive "
    #            "integer".format(self.spacing))
    #     if not isinstance(self.spacing, int):
    #         raise TypeError(msg)
    #     if self.spacing < 1:
    #         raise ValueError(msg)

    def __init__(self, arr, dim, spacing=1):
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
        # self._check_spacing()
        # self._check_arr_len()

    def _slice_arr_dim(self, slice_, arr):
        """Get a slice of a DataArray along a particular dim."""
        return arr[{self.dim: slice_}]

    def _reverse_dim(self, arr):
        """Reverse the DataArray along the given dimension."""
        return self._slice_arr_dim(slice(-1, None, -1), arr)

    def _diff(self):
        raise NotImplementedError

    def _diff_rev(self):
        raise NotImplementedError

    def diff(self):
        raise NotImplementedError

    def diff_rev(self):
        raise NotImplementedError
