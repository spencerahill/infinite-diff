"""Classes and functions for numerical analysis."""
import xarray as xr


def _check_spacing(spacing, min_spacing=1):
    """Ensure spacing value is valid."""
    msg = ("'spacing' value of {} invalid; spacing must be positive "
           "integer".format(spacing))
    if not isinstance(spacing, int):
        raise TypeError(msg)
    if spacing < min_spacing:
        raise ValueError(msg)


def _check_arr_len(arr, dim, spacing, pad=1):
    """Ensure array is long enough to perform the differencing."""
    try:
        len_arr_dim = len(arr[dim])
    except TypeError:
        len_arr_dim = 0
    if len_arr_dim < spacing + pad:
        raise ValueError("Array along dim '{}' is too small for "
                         "differencing with "
                         "spacing {}".format(dim, spacing))


class FiniteDiff(object):
    """For numerical approximations of derivatives using finite differences."""
    def __init__(self, arr, geometry='spherical',
                 vector_field=False, wraparound=False):
        """
        Create a `FiniteDiff` object.

        :param arr: Data to be finite-differenced.
        :type arr: xarray.Dataset or xarray.DataArray object
        :param str geometry: Geometry of the positions.  Either 'cartesian'
                             or 'spherical'.
        :param vector_field: Whether or not `f` is a component of a vector
                             field.  In some geometries and some directions,
                             operations differ whether the field is scalar or
                             a vector (e.g. north-south on the sphere)
        :param wraparound: Which, if any, axes are wraparound, e.g. longitude
                           on the sphere.
        """
        self.arr = arr
        self.geometry = geometry
        self.vector_field = vector_field
        self.wraparound = wraparound

    @staticmethod
    def fwd_diff(arr, dim, spacing=1):
        """Forward differencing of the array."""
        _check_spacing(spacing)
        _check_arr_len(arr, dim, spacing)
        left = arr.isel(**{dim: slice(0, -spacing)})
        right = arr.isel(**{dim: slice(spacing, None)})
        return xr.DataArray(right.values, dims=right.dims,
                            coords=left.coords) - left

    @classmethod
    def bwd_diff(cls, arr, dim, spacing=1):
        """Backward differencing of the array."""
        return -1*cls.fwd_diff(
            arr.isel(**{dim: slice(-1, None, -1)}), dim, spacing=spacing
        ).isel(**{dim: slice(-1, None, -1)})

    @classmethod
    def edges_one_sided(cls, arr, dim, spacing_left=1, spacing_right=1):
        """One-sided differencing on array edges."""
        left = arr.isel(**{dim: slice(0, spacing_left+1)})
        right = arr.isel(**{dim: slice(-(spacing_right+1), None)})
        diff_left = cls.fwd_diff(left, dim, spacing=spacing_left)
        diff_right = cls.bwd_diff(right, dim, spacing=spacing_right)
        return diff_left, diff_right

    @classmethod
    def cen_diff(cls, arr, dim, spacing=1, do_edges_one_sided=False):
        """Centered differencing of the DataArray or Dataset.

        :param arr: Data to be center-differenced.
        :type arr: `xarray.DataArray` or `xarray.Dataset`
        :param str dim: Dimension over which to perform the differencing.
        :param int spacing: How many gridpoints over to use.  Size of resulting
            array depends on this value.
        :param do_edges_one_sided: Whether or not to fill in the edge cells
            that don't have the needed neighbor cells for the stencil.  If
            `True`, use one-sided differencing with the same order of accuracy
            as `order`, and the outputted array is the same shape as `arr`.

            If `False`, the outputted array has a length in the computed axis
            reduced by `order`.
        """
        _check_spacing(spacing)
        _check_arr_len(arr, dim, 2*spacing)
        left = arr.isel(**{dim: slice(0, -spacing)})
        right = arr.isel(**{dim: slice(spacing, None)})
        diff = cls.fwd_diff(right, dim) + cls.bwd_diff(left, dim)
        if do_edges_one_sided:
            diff_left, diff_right = cls.edges_one_sided(arr, dim,
                                                        spacing_left=spacing,
                                                        spacing_right=spacing)
            diff = xr.concat([diff_left, diff, diff_right], dim=dim)
        return diff

    @staticmethod
    def arr_coord(arr, dim, coord=None):
        """Get the coord to be used as the denominator for a derivative."""
        if coord is None:
            return arr[dim]
        return coord

    @classmethod
    def fwd_diff_deriv(cls, arr, dim, coord=None, spacing=1, order=1):
        """1st order accurate forward differencing approximation of derivative.

        :param arr: Field to take derivative of.
        :param str dim: Name of dimension over which to take the derivative.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given, arr[dim] is used.
        :out: Array containing the df/dx approximation, with length in the 0th
            axis one less than that of the input array.
        """
        arr_coord = cls.arr_coord(arr, dim, coord=coord)
        if order == 1:
            return (cls.fwd_diff(arr, dim, spacing=spacing) /
                    cls.fwd_diff(arr_coord, dim, spacing=spacing))
        elif order == 2:
            # Formula is 2*fwd_diff(spacing=1) - fwd_diff(spacing=2)
            # But have to truncate fwd_diff(spacing=1) to be on same grid as
            # fwd_diff(spacing=2)
            trunc = {dim: slice(0, -spacing)}
            return (2*cls.fwd_diff_deriv(arr.isel(**trunc), dim,
                                         coord=arr_coord.isel(**trunc),
                                         spacing=spacing, order=1) -
                    cls.fwd_diff_deriv(arr, dim, coord=arr_coord,
                                       spacing=2*spacing, order=1))
        raise NotImplementedError("Forward differencing derivative only "
                                  "supported for 1st and 2nd order currently")

    @classmethod
    def bwd_diff_deriv(cls, arr, dim, coord=None, spacing=1, order=1):
        """1st order accurate backward differencing approx of derivative.

        :param arr: Field to take derivative of.
        :param str dim: Name of dimension over which to take the derivative.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given, arr[dim] is used.
        :out: Array containing the df/dx approximation, with length in the 0th
              axis one less than that of the input array.
        """
        return cls.fwd_diff_deriv(
            arr.isel(**{dim: slice(-1, None, -1)}), dim, coord=coord,
            spacing=spacing, order=order
        ).isel(**{dim: slice(-1, None, -1)})

    @classmethod
    def cen_diff_deriv(cls, arr, dim, coord=None, order=2,
                       do_edges_one_sided=False):
        """
        Centered differencing approximation of 1st derivative.

        :param arr: Data to be center-differenced.
        :type arr: `xarray.DataArray` or `xarray.Dataset`
        :param str dim: Dimension over which to compute.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given or None, arr[dim] is used.
        :param int order: Order of accuracy to use.  Default 2.
        :param do_edges_one_sided: Whether or not to fill in the edge cells
            that don't have the needed neighbor cells for the stencil.  If
            `True`, use one-sided differencing with the same order of accuracy
            as `order`, and the outputted array is the same shape as `arr`.

            If `False`, the outputted array has a length in the computed axis
            reduced by `order`.
        """
        if order != 2:
            raise NotImplementedError("Centered differencing of df/dx only "
                                      "supported for 2nd order currently")
        if coord is None:
            arr_coord = arr[dim]
        else:
            arr_coord = coord
        numer = cls.cen_diff(arr, dim, spacing=1,
                             do_edges_one_sided=do_edges_one_sided)
        denom = cls.cen_diff(arr_coord, dim, spacing=1,
                             do_edges_one_sided=do_edges_one_sided)
        return numer / denom

    @classmethod
    def upwind_advec(cls, arr, flow, dim, coord=None, order=1,
                     wraparound=False):
        """
        Upwind differencing scheme for advection.

        :param arr: Field being advected.
        :param flow: Flow that is advecting the field.
        """
        if order == 2:
            return cls.upwind_advec2(arr, flow, dim, coord=coord,
                                     wraparound=wraparound)
        flow_pos = flow.copy()
        flow_pos.values[flow.values < 0] = 0.
        flow_neg = flow.copy()
        flow_neg.values[flow.values >= 0] = 0.
        fwd = cls.bwd_diff_deriv(arr, dim, coord=coord, order=order)
        bwd = cls.fwd_diff_deriv(arr, dim, coord=coord, order=order)
        interior = flow_pos*bwd + flow_neg*fwd
        # If array has wraparound values, no special edge handling needed.
        if wraparound:
            return interior
        # Edge cases can't do upwind.
        slice_left = {dim: slice(0, order+1)}
        slice_right = {dim: slice(-(order+1), None)}
        if coord is None:
            coord_left = None
            coord_right = None
        else:
            coord_left = coord.isel(**slice_left)
            coord_right = coord.isel(**slice_right)
        left_edge = flow.isel(**{dim: 0}) * cls.fwd_diff_deriv(
            arr.isel(**slice_left), dim, coord=coord_left, order=order
        )
        right_edge = flow.isel(**{dim: -1}) * cls.bwd_diff_deriv(
            arr.isel(**slice_right), dim, coord=coord_right, order=order
        )
        return xr.concat([left_edge, interior, right_edge], dim=dim)

    @classmethod
    def upwind_advec2(cls, arr, flow, dim, coord=None, wraparound=False):
        """
        Upwind differencing scheme for advection with 2nd order accuracy.

        :param arr: Field being advected.
        :param flow: Flow that is advecting the field.
        """
        flow_pos = flow.copy()
        flow_pos.values[flow.values < 0] = 0.
        flow_neg = flow.copy()
        flow_neg.values[flow.values >= 0] = 0.
        fwd = cls.bwd_diff2_deriv(arr, dim, coord=coord)
        bwd = cls.fwd_diff2_deriv(arr, dim, coord=coord)
        interior = flow_pos*bwd + flow_neg*fwd
        # If array has wraparound values, no special edge handling needed.
        if wraparound:
            return interior
        # Edge cases can't do upwind.
        order = 2
        slice_left = {dim: slice(0, order)}
        slice_right = {dim: slice(-order, None)}
        if coord is None:
            coord_left = None
            coord_right = None
        else:
            coord_left = coord.isel(**slice_left)
            coord_right = coord.isel(**slice_right)
        left_edge = flow.isel(**slice_left) * cls.fwd_diff_deriv(
            arr.isel(**slice_left), dim, coord=coord_left
        )
        right_edge = flow.isel(**slice_right) * cls.bwd_diff_deriv(
            arr.isel(**slice_right), dim, coord=coord_right
        )
        return xr.concat([left_edge, interior, right_edge], dim=dim)
