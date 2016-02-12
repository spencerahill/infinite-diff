"""Vertically oriented coordinates."""
from . import Coord


class VertCoord(Coord):
    """Base class for vertical coordinates."""
    def __init__(self, *args, **kwargs):
        arr = args[0]
        dim = kwargs.get('dim', None)
        super(VertCoord, self).__init__(arr, dim=dim, cyclic=False)


class ZCoord(VertCoord):
    """Height vertical coordinates."""
    def __init__(self, *args, **kwargs):
        z = args[0]
        dim = kwargs.get('dim', None)
        super(ZCoord, self).__init__(z, dim=dim)


class Pressure(VertCoord):
    """Pressure vertical coordinates."""
    def __init__(self, *args, **kwargs):
        p = args[0]
        dim = kwargs.get('dim', None)
        super(Pressure, self).__init__(p, dim=dim)


class Sigma(VertCoord):
    """Pressure divided by surface pressure vertical coordinates."""
    def __init__(self, *args, **kwargs):
        sigma = args[0]
        dim = kwargs.get('dim', None)
        super(Sigma, self).__init__(sigma, dim=dim)

    def pressure(self, ps):
        """Get pressure from sigma levels and surface pressure."""
        return ps * self.sigma


class Eta(VertCoord):
    """Hybrid sigma-pressure vertical coordinates."""
    def __init__(self, *args, **kwargs):
        p_ref = args[0]
        self.bk = args[1]
        self.pk = args[2]
        dim = kwargs.get('dim', None)
        super(Eta, self).__init__(p_ref, dim=dim)

    def pressure(self, ps):
        """Compute pressure from surface pressure and eta coordinate arrays."""
        return self.pk + self.bk*ps
