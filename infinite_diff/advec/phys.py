from .._constants import LON_STR, LAT_STR
from ..deriv import (PhysDeriv, LonBwdDeriv, LonFwdDeriv, LatBwdDeriv,
                     LatFwdDeriv, EtaBwdDeriv, EtaFwdDeriv,
                     SphereEtaBwdDeriv, SphereEtaFwdDeriv)
from ..geom import SphereEtaGeom
from . import Upwind


class PhysUpwind(Upwind):
    """Upwind advection along a physical coordinate."""
    _DERIV_BWD_CLS = PhysDeriv
    _DERIV_FWD_CLS = PhysDeriv
    _ADVEC_CLS = Upwind

    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True, cyclic=False):
        super(PhysUpwind, self).__init__(flow, arr, dim, coord=coord,
                                         spacing=spacing, order=order,
                                         fill_edge=fill_edge)
        self._deriv_bwd_obj = self._DERIV_BWD_CLS(
            arr, dim, coord=coord, spacing=spacing, order=order,
            fill_edge=fill_edge, cyclic=cyclic
        )
        self._deriv_fwd_obj = self._DERIV_FWD_CLS(
            arr, dim, coord=coord, spacing=spacing, order=order,
            fill_edge=fill_edge, cyclic=cyclic
        )
        self._deriv_bwd = self._deriv_bwd_obj.deriv
        self._deriv_fwd = self._deriv_fwd_obj.deriv


class LonUpwind(PhysUpwind):
    """Upwind advection in longitude."""
    _DERIV_BWD_CLS = LonBwdDeriv
    _DERIV_FWD_CLS = LonFwdDeriv

    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True, cyclic=False):
        super(PhysUpwind, self).__init__(flow, arr, dim, coord=coord,
                                         spacing=spacing, order=order,
                                         fill_edge=fill_edge)
        self._deriv_bwd_obj = self._DERIV_BWD_CLS(
            arr, dim, coord=coord, spacing=spacing, order=order,
            fill_edge=fill_edge, cyclic=cyclic
        )
        self._deriv_fwd_obj = self._DERIV_FWD_CLS(
            arr, dim, coord=coord, spacing=spacing, order=order,
            fill_edge=fill_edge, cyclic=cyclic
        )
        self._deriv_bwd = self._deriv_bwd_obj.deriv
        self._deriv_fwd = self._deriv_fwd_obj.deriv


class LatUpwind(PhysUpwind):
    """Upwind advection in latitude."""
    _DERIV_BWD_CLS = LatBwdDeriv
    _DERIV_FWD_CLS = LatFwdDeriv


class EtaUpwind(PhysUpwind):
    """Vertical upwind advection in hybrid pressure-sigma coordinates."""
    _DERIV_BWD_CLS = EtaBwdDeriv
    _DERIV_FWD_CLS = EtaFwdDeriv


class SphereEtaUpwind(object):
    _ADVEC_CLS = Upwind
    _DERIV_BWD_CLS = SphereEtaBwdDeriv
    _DERIV_FWD_CLS = SphereEtaFwdDeriv

    def __init__(self, arr, pk, bk, pfull, **kwargs):
        self._arr = arr
        self._pk = pk
        self._bk = bk
        self._geom = SphereEtaGeom(arr[LON_STR], arr[LAT_STR], pk, bk, pfull,
                                   **kwargs)
        self._x = self._geom.x
        self._y = self._geom.y
        self._z = self._geom.z

    def advec_x(self, u, **kwargs):
        return self._ADVEC_CLS(u, self._arr, LON_STR, **kwargs).advec()

    def advec_y(self, v, **kwargs):
        pass

    def advec_z(self, omega, **kwargs):
        pass
