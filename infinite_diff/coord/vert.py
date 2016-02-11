"""Vertically oriented coordinates."""
from . import Coord


class VertCoord(Coord):
    """Base class for vertical coordinates."""
    def __init__(self, arr, dim=None):
        super(VertCoord, self).__init__(arr, dim=dim, cyclic=False)


class ZCoord(VertCoord):
    """Height vertical coordinates."""
    def __init__(self, z, dim=None):
        super(ZCoord, self).__init__(z, dim=dim)


class Pressure(VertCoord):
    """Pressure vertical coordinates."""
    def __init__(self, p, dim=None):
        super(Pressure, self).__init__(p, dim=dim)


class Sigma(VertCoord):
    """Pressure divided by surface pressure vertical coordinates."""
    def __init__(self, p, dim=None):
        super(Sigma, self).__init__(p, dim=dim)


class Eta(VertCoord):
    """Hybrid sigma-pressure vertical coordinates."""
    def __init__(self, p_ref, pk, bk, dim=None):
        super(Eta, self).__init__(p_ref, dim=dim)
        self.bk = bk
        self.pk = pk

    def pressure(self, ps):
        """Compute pressure from surface pressure."""
        return self.pk + self.bk*ps
