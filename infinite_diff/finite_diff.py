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

    @staticmethod
    def reverse_dim(arr, dim):
        """Reverse the DataArray along the given dimension."""
        return arr.isel(**{dim: slice(-1, None, -1)})

    @classmethod
    def bwd_diff(cls, arr, dim, spacing=1):
        """Backward differencing of the array."""
        return -1*cls.reverse_dim(
            cls.fwd_diff(cls.reverse_dim(arr, dim), dim, spacing=spacing), dim
        )

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
    def fwd_diff_deriv(cls, arr, dim, coord=None, spacing=1, order=1,
                       fill_edge=False):
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
            interior = (cls.fwd_diff(arr, dim, spacing=spacing) /
                        cls.fwd_diff(arr_coord, dim, spacing=spacing))
            if not fill_edge:
                return interior
            right_slice = {dim: slice(-(spacing+1), None)}
            right_edge = (cls.bwd_diff(arr.isel(**right_slice), dim,
                                       spacing=spacing) /
                          cls.bwd_diff(arr_coord.isel(**right_slice), dim,
                                       spacing=spacing))
            return xr.concat([interior, right_edge], dim=dim)
        elif order == 2:
            single_space = cls.fwd_diff_deriv(arr, dim, coord=arr_coord,
                                              spacing=spacing, order=1,
                                              fill_edge=fill_edge)
            double_space = cls.fwd_diff_deriv(arr, dim, coord=arr_coord,
                                              spacing=2*spacing, order=1,
                                              fill_edge=False)
            interior = 2*single_space - double_space
            if not fill_edge:
                return interior
            right_edge = {dim: slice(-order*spacing, None)}
            return xr.concat([interior, single_space.isel(**right_edge)],
                             dim=dim)
        raise NotImplementedError("Forward differencing derivative only "
                                  "supported for 1st and 2nd order currently")

    @classmethod
    def bwd_diff_deriv(cls, arr, dim, coord=None, spacing=1, order=1,
                       fill_edge=False):
        """1st order accurate backward differencing approx of derivative.

        :param arr: Field to take derivative of.
        :param str dim: Name of dimension over which to take the derivative.
        :param xarray.DataArray coord: Coordinate array to use for the
            denominator.  If not given, arr[dim] is used.
        :out: Array containing the df/dx approximation, with length in the 0th
              axis one less than that of the input array.
        """
        arr_coord = cls.arr_coord(arr, dim, coord=coord)
        return cls.reverse_dim(
            cls.fwd_diff_deriv(cls.reverse_dim(arr, dim), dim,
                               coord=cls.reverse_dim(arr_coord, dim),
                               spacing=spacing, order=order,
                               fill_edge=fill_edge), dim
        )

    @classmethod
    def cen_diff_deriv(cls, arr, dim, coord=None, spacing=1, order=2,
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
        arr_coord = cls.arr_coord(arr, dim, coord=coord)
        if order == 2:
            numer = cls.cen_diff(arr, dim, spacing=spacing,
                                 do_edges_one_sided=do_edges_one_sided)
            denom = cls.cen_diff(arr_coord, dim, spacing=spacing,
                                 do_edges_one_sided=do_edges_one_sided)
            return numer / denom
        elif order == 4:
            # Formula is (4/3)*cen_diff(spacing=1) - (1/3)*cen_diff(spacing=2)
            # But have to truncate cen_diff(spacing=1) to be on same grid as
            # cen_diff(spacing=2)
            trunc = {dim: slice(spacing, -spacing)}
            return (4*cls.cen_diff_deriv(
                arr.isel(**trunc), dim, coord=arr_coord.isel(**trunc),
                spacing=spacing, order=2, do_edges_one_sided=do_edges_one_sided
            ) - cls.cen_diff_deriv(
                arr, dim, coord=arr_coord, spacing=2*spacing, order=2,
                do_edges_one_sided=do_edges_one_sided
            )) / 3.
        raise NotImplementedError("Centered differencing only "
                                  "supported for 2nd and 4th order.")

    @staticmethod
    def upwind_advec_flow(flow, reverse_dim=False):
        """Create negative- and positive-only arrays for upwind advection.

        :param flow: Flow that is advecting some field.
        :type flow: xarray.DataArray
        :param reverse_dim: `True` signifies that the dimension in question has
            a physical orientation opposite to its orientation within the
            DataArray.  E.g. if the flow is the pressure velocity, omega, and
            the pressure coordinate goes from large to small pressure values,
            then this flag shold be set to `True`.  Default is `False`.
        :type reverse_dim: bool
        :out: flow_neg, flow_pos xarray.DataArrays with shape and coords
            identical to `flow1, but with, respectively, all positive and
            negative values set to 0 (or the reverse if `reverse_dim` is
            `True`).
        """
        flow_neg = flow.copy()
        flow_neg.values[flow.values >= 0] = 0.
        flow_pos = flow.copy()
        flow_pos.values[flow.values < 0] = 0.
        if not reverse_dim:
            return flow_neg, flow_pos
        return flow_pos, flow_neg

    @classmethod
    def upwind_advec(cls, arr, flow, dim, coord=None, spacing=1, order=1,
                     reverse_dim=False, fill_edge=True):
        """
        Upwind differencing scheme for advection.

        :param arr: Field being advected.
        :param flow: Flow that is advecting the field.
        """
        fwd = cls.fwd_diff_deriv(arr, dim, coord=coord, spacing=spacing,
                                 order=order, fill_edge=True)
        bwd = cls.bwd_diff_deriv(arr, dim, coord=coord, spacing=spacing,
                                 order=order, fill_edge=True)
        # Forward diff on left edge; backward diff on right edge.
        edge_right = {dim: -1}
        edge_left = {dim: 0}
        fwd[edge_right] = bwd[edge_right]
        bwd[edge_left] = fwd[edge_left]
        # In interior, forward diff for negative flow, backward for positive.
        neg, pos = cls.upwind_advec_flow(flow, reverse_dim=reverse_dim)
        advec = pos*bwd + neg*fwd
        # Truncate to interior edges of filled edges not desired.
        if not fill_edge:
            slice_middle = {dim: slice(order, -order)}
            advec = advec.isel(**slice_middle)
        return advec
