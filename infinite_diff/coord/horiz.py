from aospy.utils import to_radians
import numpy as np
import xarray as xr

from .. import _RADEARTH
from . import Coord


def wraparound(arr, dim, left=1, right=1, circumf=360., spacing=1):
    """Append wrap-around point(s) to the DataArray or Dataset coord."""
    if left:
        edge_left = arr.isel(**{dim: slice(0, left, spacing)})
        edge_left[dim] += circumf
        arr = xr.concat([arr, edge_left], dim=dim)
    if right:
        edge_right = arr.isel(**{dim: slice(-right, None, spacing)})
        edge_right[dim] -= circumf
        xr.concat([edge_right, arr], dim=dim)
    return arr


class HorizCoord(Coord):
    """General horizontally oriented coordinate."""
    def __init__(self, arr, dim=None, cyclic=False):
        super(HorizCoord, self).__init__(arr, dim=dim, cyclic=cyclic)
        self.cyclic = cyclic


class XCoord(HorizCoord):
    """Cartesian x horizontal coordinate."""
    def __init__(self, arr, dim=None, cyclic=False):
        super(XCoord, self).__init__(arr, dim=dim, cyclic=cyclic)

    def deriv_prefactor(self, *args, **kwargs):
        """Factor to multiply the result of derivatives along this coord."""
        return 1.

    def deriv_factor(self, *args, **kwargs):
        """Factor to multiply within a derivative along this coord."""
        return 1.


class YCoord(HorizCoord):
    """Cartesian y horizontal coordinate."""
    def __init__(self, arr, dim=None, cyclic=False):
        super(YCoord, self).__init__(arr, dim=dim, cyclic=cyclic)

    def deriv_prefactor(self, *args, **kwargs):
        """Factor to multiply the result of derivatives along this coord."""
        return 1.

    def deriv_factor(self, *args, **kwargs):
        """Factor to multiply within a derivative along this coord."""
        return 1.


class Lon(XCoord):
    """Longitude spherical horizontal coordinate."""
    def __init__(self, lon, dim=None, cyclic=True, radius=_RADEARTH):
        super(Lon, self).__init__(lon, dim=dim, cyclic=cyclic)
        self.radius = radius
        self._lon_rad = to_radians(lon)

    def deriv_prefactor(self, lat):
        return 1. / (self.radius * np.cos(to_radians(lat)))


class Lat(YCoord):
    """Latitude spherical horizontal coordinate."""
    def __init__(self, lat, dim=None, cyclic=False, radius=_RADEARTH):
        super(Lat, self).__init__(lat, dim=dim, cyclic=cyclic)
        self.radius = radius
        self._lat_rad = to_radians(lat)

    def deriv_prefactor(self, oper='grad'):
        if oper == 'grad':
            return 1. / self.radius
        if oper == 'divg':
            return 1. / (self.radius * np.cos(self._lat_rad))

    def deriv_factor(self):
        return 1. / (self.radius * np.cos(self._lat_rad))
