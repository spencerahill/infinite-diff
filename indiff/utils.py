"""Utility functions."""
import warnings

import numpy as np
import xarray as xr


def replace_coord(arr, old_dim, new_dim, new_coord):
    """Replace a coordinate with new one; new and old must have same shape."""
    new_arr = arr.rename({old_dim: new_dim})
    ds = new_arr.to_dataset(name='new_arr')
    ds[new_dim] = new_coord
    return ds['new_arr']


def _arr_deep_copy(arr):
    arr_copy = arr.copy(deep=True)
    arr_props = {prop: getattr(arr_copy, prop) for prop in
                 ['name', 'attrs', 'encoding']}
    return xr.DataArray(arr_copy.values.copy(), dims=arr_copy.dims,
                        coords=arr_copy.coords.to_dataset().copy(deep=True),
                        **arr_props)


def deep_copy(arr):
    """Create a copy of an array fully uncoupled from the original.

    This is more thorough than xarray's copy(deep=True) method.
    """
    if isinstance(arr, (float, int)):
        return arr
    arr_copy = arr.copy(deep=True)
    arr_values = arr_copy.values.copy()
    new_dims = arr_copy.dims
    coords = arr_copy.coords.to_dataset().copy(deep=True)
    new_coords = {key: _arr_deep_copy(val) for key, val in coords.items()}
    arr_props = {prop: getattr(arr_copy, prop) for prop in
                 ['name', 'attrs', 'encoding']}
    return xr.DataArray(arr_values, dims=new_dims, coords=new_coords,
                        **arr_props)


def to_radians(arr, is_delta=False):
    """Force data with units either degrees or radians to be radians."""
    # Infer the units from embedded metadata, if it's there.
    arr_out = arr
    try:
        units = arr_out.units
    except AttributeError:
        pass
    else:
        if units.lower().startswith('degrees'):
            warn_msg = ("Conversion applied: degrees -> radians to arr_outay: "
                        "{}".format(arr_out))
            warnings.warn(warn_msg, UserWarning)
            return np.deg2rad(arr_out)
    # Otherwise, assume degrees if the values are sufficiently large.
    threshold = 0.1*np.pi if is_delta else 4*np.pi
    if np.max(np.abs(arr_out)) > threshold:
        warn_msg = ("Conversion applied: degrees -> radians to arr_outay: "
                    "{}".format(arr_out))
        warnings.warn(warn_msg, UserWarning)
        return np.deg2rad(arr_out)
    return arr_out


def add_cyclic_to_left(arr, dim, num_points, circumf):
    if not num_points:
        return arr
    # Make an isolated copy of the original coord values.
    arr_dim_values = arr[dim].values.copy()
    # Make an isolated copy of the whole original DataArray.
    arr_out = xr.DataArray(arr.values.copy(),
                           dims=arr.dims, coords=arr.coords)
    # Grab the edge values of the copied array.
    trunc = {dim: slice(-num_points, None)}
    edge = deep_copy(arr_out.isel(**trunc))
    # Subtract the circumference from the coordinates
    edge[dim] -= circumf
    # Join together the modified edge coord with the original coord.
    new_coord_values = np.concatenate([edge[dim].values.copy(),
                                       arr_dim_values])
    # Join together the edge array with the original array.
    new_arr = xr.concat([edge, arr], dim=dim)
    # Override the coord with the one just created.
    new_arr[dim].values = new_coord_values
    return new_arr


def add_cyclic_to_right(arr, dim, num_points, circumf):
    if not num_points:
        return arr
    # Make an isolated copy of the original coord values.
    arr_dim_values = arr[dim].values.copy()
    # Make an isolated copy of the whole original DataArray.
    arr_out = xr.DataArray(arr.values.copy(),
                           dims=arr.dims, coords=arr.coords)
    # Grab the edge values of the copied array.
    trunc = {dim: slice(0, num_points)}
    edge = deep_copy(arr_out.isel(**trunc))
    # Add the circumference to the coordinates.
    edge[dim] += circumf
    # Join together the modified edge coord with the original coord.
    new_coord_values = np.concatenate([arr_dim_values,
                                       edge[dim].values.copy()])
    # Join together the edge array with the original array.
    new_arr = xr.concat([arr, edge], dim=dim)
    # Override the coord with the one just created.
    new_arr[dim].values = new_coord_values
    return new_arr


def wraparound(arr, dim, left_to_right=0, right_to_left=0,
               circumf=360., spacing=1):
    """Append wrap-around point(s) to the DataArray or Dataset coord."""
    # TODO: Bugfix: if points on both sides are added, as written this
    # causes the points first added to the right to then be added to the left,
    # rather than the original points on the right.
    new = add_cyclic_to_right(arr, dim, left_to_right, circumf)
    return add_cyclic_to_left(new, dim, right_to_left, circumf)
