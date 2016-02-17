
import numpy as np

from .._constants import _RADEARTH
from ..utils import to_radians
from . import Coord


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
    _POSSIBLY_CYCLIC = False

    def __init__(self, lat, dim=None, radius=_RADEARTH):
        super(Lat, self).__init__(lat, dim=dim, cyclic=False)
        self.radius = radius
        self._lat_rad = to_radians(lat)

    def deriv_prefactor(self, oper='grad'):
        if oper == 'grad':
            return 1. / self.radius
        if oper == 'divg':
            return 1. / (self.radius * np.cos(self._lat_rad))
        msg = "'oper' must be 'grad' or 'divg': value was '{}'".format(oper)
        raise ValueError(msg)

    def deriv_factor(self, oper='grad'):
        if oper == 'grad':
            return 1.
        if oper == 'divg':
            return np.cos(self._lat_rad)
        msg = "'oper' must be 'grad' or 'divg': value was '{}'".format(oper)
        raise ValueError(msg)
