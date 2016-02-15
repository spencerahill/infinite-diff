from ..diff import FiniteDiff


class FiniteDeriv(object):
    """Base class for finite-diff based derivative classes."""
    _DIFF_CLS = FiniteDiff

    def _arr_coord(self, coord):
        """Get the coord to be used as the denominator for a derivative."""
        if coord is None:
            return self.arr[self.dim]
        return coord

    def __init__(self, arr, dim, coord=None):
        """
        :param arr: Field to take derivative of.
        :param str dim: Name of dimension over which to take the derivative.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given, arr[dim] is used.
        """
        self.arr = arr
        self.dim = dim
        self.coord = self._arr_coord(coord)

        self._arr_diff_obj = self._DIFF_CLS(arr, dim)
        self._coord_diff_obj = self._DIFF_CLS(self.coord, dim)
        self._arr_diff = self._arr_diff_obj.diff
        self._coord_diff = self._coord_diff_obj.diff

    def _deriv(self, spacing=1):
        """Core finite-differencing derivative; no edge handling."""
        return (self._arr_diff(spacing=spacing) /
                self._coord_diff(spacing=spacing))

    def _slice_edge(self, arr, spacing, order, pad=1):
        raise NotImplementedError

    def _concat(self):
        raise NotImplementedError

    def deriv(self, spacing=1, order=1, fill_edge=True):
        raise NotImplementedError
