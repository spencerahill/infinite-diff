"""Vertically oriented coordinates."""
from .._constants import PHALF_STR, PFULL_STR
from ..utils import replace_coord
from ..diff import CenDiff
from . import Coord


class VertCoord(Coord):
    """Base class for vertical coordinates."""
    _POSSIBLY_CYCLIC = False

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
    def __init__(self, sigma, dim=None):
        super(Sigma, self).__init__(sigma, dim=dim)

    def pressure(self, ps):
        """Get pressure from sigma levels and surface pressure."""
        return ps * self.sigma


class Eta(VertCoord):
    """Hybrid sigma-pressure vertical coordinates."""
    def __init__(self, pk, bk, pfull, dim=None):
        self.pk = pk
        self.bk = bk
        self.pfull = pfull
        self.phalf = self.pk[PHALF_STR]
        self.arr = self.phalf
        self.dim = dim if dim is not None else self.pk.dims[0]

    def phalf_from_ps(self, ps):
        """Compute pressure at level edges from surface pressure."""
        return self.pk + self.bk*ps

    def to_pfull_from_phalf(self, arr):
        """Compute data at full pressure levels from values at half levels."""
        arr_top = arr.copy()[{PHALF_STR: slice(1, None)}]
        arr_top = replace_coord(arr_top, PHALF_STR, PFULL_STR, self.pfull)

        arr_bot = arr.copy()[{PHALF_STR: slice(None, -1)}]
        arr_bot = replace_coord(arr_bot, PHALF_STR, PFULL_STR, self.pfull)
        return 0.5*(arr_bot + arr_top)

    def pfull_from_ps(self, ps):
        """Compute pressure at full levels from surface pressure."""
        return self.to_pfull_from_phalf(self.phalf_from_ps(ps))

    def d_deta_from_phalf(self, arr):
        """Compute pressure level thickness from half level pressures."""
        d_deta = arr.diff(dim=PHALF_STR, n=1)
        return replace_coord(d_deta, PHALF_STR, PFULL_STR, self.pfull)

    def d_deta_from_pfull(self, arr):
        """Compute $\partial/\partial\eta$ of the array on full hybrid levels.

        $\eta$ is the model vertical coordinate, and its value is assumed to
        simply increment by 1 from 0 at the surface upwards.  The data to be
        differenced is assumed to be defined at full pressure levels.
        """
        deriv = CenDiff(arr, PFULL_STR, spacing=1, fill_edge=True).diff() / 2.
        # Edges use 1-sided differencing, so only spanning one level, not two.
        deriv[{PFULL_STR: 0}] = deriv[{PFULL_STR: 0}] * 2.
        deriv[{PFULL_STR: -1}] = deriv[{PFULL_STR: -1}] * 2.
        return deriv

    def dp_from_ps(self, ps):
        """Compute pressure level thickness from surface pressure"""
        return self.d_deta_from_phalf(self.phalf_from_ps(ps))
