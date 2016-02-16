"""Horizontal geometries."""
from ..coord import HorizCoord, XCoord, YCoord, Lon, Lat


class HorizGeom(object):
    """Generic base class for horizontal geometries."""
    _X_COORD_CLS = HorizCoord
    _Y_COORD_CLS = HorizCoord
    _X_DIM_NAME = 'x'
    _Y_DIM_NAME = 'y'

    def _prep_coord(self, coord, coord_cls, dim):
        if isinstance(coord, coord_cls):
            return coord
        return coord_cls(coord, dim)

    def __init__(self, x, y):
        self.x = self._prep_coord(x, self._X_COORD_CLS, self._X_DIM_NAME)
        self.y = self._prep_coord(y, self._Y_COORD_CLS, self._Y_DIM_NAME)
        self._x_arr = self.x.arr
        self._y_arr = self.y.arr


class HorizCartesian(HorizGeom):
    """Cartesian horizontal geometry."""
    _X_COORD_CLS = XCoord
    _Y_COORD_CLS = YCoord

    def __init__(self, x, y):
        super(HorizCartesian, self).__init__(x, y)


class HorizSphere(HorizGeom):
    """Spherical horizontal geometry."""
    _X_COORD_CLS = Lon
    _Y_COORD_CLS = Lat
    _X_DIM_NAME = 'lon'
    _Y_DIM_NAME = 'lat'

    def __init__(self, lon, lat):
        super(HorizSphere, self).__init__(lon, lat)
