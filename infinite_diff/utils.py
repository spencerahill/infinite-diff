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


def to_radians(arr, is_delta=False):
    """Force data with units either degrees or radians to be radians."""
    # Infer the units from embedded metadata, if it's there.
    try:
        units = arr.units
    except AttributeError:
        pass
    else:
        if units.lower().startswith('degrees'):
            warn_msg = ("Conversion applied: degrees -> radians to array: "
                        "{}".format(arr))
            warnings.warn(warn_msg, UserWarning)
            return np.deg2rad(arr)
    # Otherwise, assume degrees if the values are sufficiently large.
    threshold = 0.1*np.pi if is_delta else 4*np.pi
    if np.max(np.abs(arr)) > threshold:
        warn_msg = ("Conversion applied: degrees -> radians to array: "
                    "{}".format(arr))
        warnings.warn(warn_msg, UserWarning)
        return np.deg2rad(arr)
    return arr


def wraparound(arr, dim, left_to_right=0, right_to_left=0,
               circumf=360., spacing=1):
    """Append wrap-around point(s) to the DataArray or Dataset coord."""
    arr_out = arr.copy()
    if left_to_right:
        edge_left = arr.copy()[{dim: slice(0, left_to_right, spacing)}]
        edge_left[dim] += circumf
        arr_out = xr.concat([arr_out, edge_left], dim=dim)
    if right_to_left:
        edge_right = arr.copy()[{dim: slice(-right_to_left, None, spacing)}]
        edge_right[dim] -= circumf
        arr_out = xr.concat([edge_right, arr_out], dim=dim)
    return arr_out
