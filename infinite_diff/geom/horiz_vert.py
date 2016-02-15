"""Combined horizontal geometries with vertical coordinates."""
from .. import _RADEARTH
from ..coord import ZCoord, Pressure, Eta
from . import HorizCartesian, HorizSphere


class HorizVertGeom(object):
    def __init__(self, horiz_geom, vert_coord):
        self._horiz_geom = horiz_geom
        self._vert_coord = vert_coord
        self.x = self._horiz_geom.x
        self.y = self._horiz_geom.y
        self.z = self._vert_coord
        self.d_dx = self._horiz.d_dx
        self.d_dy = self._horiz.d_dy
        self.horiz_grad = self._horiz_geom.grad


class Cartesian3D(HorizVertGeom):
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
        super(Cartesian3D, self).__init__(HorizCartesian(x, y), ZCoord(z))


class SpherePressureGeom(HorizVertGeom):
    def __init__(self, lon, lat, p, radius=_RADEARTH):
        super(SphereEtaGeom, self).__init__(HorizSphere(lon, lat),
                                            Pressure(p))


class SphereEtaGeom(HorizVertGeom):
    def __init__(self, lon, lat, pk, bk, radius=_RADEARTH):
        super(SphereEtaGeom, self).__init__(HorizSphere(lon, lat),
                                            Eta(pk, bk))
