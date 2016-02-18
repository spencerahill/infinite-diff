import xarray as xr

from .._constants import LON_STR, LAT_STR, PFULL_STR
from ..utils import deep_copy
from ..deriv import (PhysDeriv, LonBwdDeriv, LonFwdDeriv, LatBwdDeriv,
                     LatFwdDeriv, EtaBwdDeriv, EtaFwdDeriv,
                     SphereEtaBwdDeriv, SphereEtaFwdDeriv)
from . import Upwind


def _make_derivs(obj, arr, *deriv_args, **deriv_kwargs):
    b = deep_copy(arr)
    f = deep_copy(arr)
    obj._deriv_bwd_obj = obj._DERIV_BWD_CLS(b, *deriv_args, **deriv_kwargs)
    obj._deriv_bwd = getattr(obj._deriv_bwd_obj, obj._DERIV_METHOD)
    obj._deriv_fwd_obj = obj._DERIV_FWD_CLS(f, *deriv_args, **deriv_kwargs)
    obj._deriv_fwd = getattr(obj._deriv_fwd_obj, obj._DERIV_METHOD)


class PhysUpwind(Upwind):
    """Upwind advection along a physical coordinate."""
    _DERIV_BWD_CLS = PhysDeriv
    _DERIV_FWD_CLS = PhysDeriv
    _DERIV_METHOD = 'deriv'
    _ADVEC_CLS = Upwind

    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 cyclic=False, fill_edge=True):
        self.flow = flow
        self.arr = arr.copy(deep=True)
        self.dim = dim
        self.coord = coord
        self.spacing = spacing
        self.order = order
        self.cyclic = cyclic
        self.fill_edge = fill_edge

        deriv_args = [dim]
        deriv_kwargs = dict(coord=coord, spacing=spacing, order=order,
                            cyclic=cyclic, fill_edge=fill_edge)
        _make_derivs(self, arr.copy(deep=True), *deriv_args, **deriv_kwargs)

    def _derivs_bwd_fwd(self, *args, **kwargs):
        """Generate forward and backward differencing derivs for upwind.

        If not cyclic, order of accuracy decreases moving towards edge (right
        edge for forward, left edge for backward) as the differencing stencil
        starts to extend over the domain edge.  At the edge itself, the
        opposite signed differencing is used with the same order of accuracy as
        in the interior.
        """
        bwd = self._deriv_bwd(*args, **kwargs)
        fwd = self._deriv_fwd(*args, **kwargs)
        # Forward diff on left edge; backward diff on right edge if not cyclic.
        if getattr(self, 'cyclic', False):
            return bwd, fwd
        return self._swap_bwd_fwd_edges(bwd, fwd)

    def advec(self, *args, **kwargs):
        """
        Upwind differencing scheme for advection.

        In interior, forward differencing for negative flow, and backward
        differencing for positive flow.

        :param arr: Field being advected.
        :param flow: Flow that is advecting the field.
        """
        bwd, fwd = self._derivs_bwd_fwd(*args, **kwargs)
        neg, pos = self._flow_neg_pos()
        advec_arr = pos*bwd + neg*fwd
        if not self.fill_edge and not self.cyclic:
            slice_middle = {self.dim: slice(self.order, -self.order)}
            advec_arr = advec_arr[slice_middle]
        return advec_arr


class LonUpwind(PhysUpwind):
    """Upwind advection in longitude."""
    _DERIV_BWD_CLS = LonBwdDeriv
    _DERIV_FWD_CLS = LonFwdDeriv
    _DIM = LON_STR

    def __init__(self, flow, arr, dim=None, coord=None, spacing=1, order=2,
                 cyclic=True, fill_edge=False):
        self.flow = flow
        self.arr = arr.copy(deep=True)
        self.spacing = spacing
        self.order = order
        self.cyclic = cyclic
        self.fill_edge = fill_edge

        self.dim = dim if dim is not None else self._DIM
        self.coord = coord if coord is not None else self.arr[self._DIM]

        deriv_args = [self.dim]
        deriv_kwargs = dict(coord=coord, spacing=spacing, order=order,
                            fill_edge=True, cyclic=cyclic)
        _make_derivs(self, arr.copy(deep=True), *deriv_args, **deriv_kwargs)


class LatUpwind(PhysUpwind):
    """Upwind advection in latitude."""
    _DERIV_BWD_CLS = LatBwdDeriv
    _DERIV_FWD_CLS = LatFwdDeriv
    _DIM = LAT_STR

    def __init__(self, flow, arr, dim=None, coord=None, spacing=1, order=2,
                 fill_edge=True):
        self.flow = flow
        self.arr = arr.copy(deep=True)
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge

        self.dim = dim if dim is not None else self._DIM
        self.coord = coord if coord is not None else self.arr[self._DIM]

        deriv_args = [self.dim]
        deriv_kwargs = dict(coord=coord, spacing=spacing, order=order,
                            fill_edge=True)
        _make_derivs(self, arr.copy(deep=True), *deriv_args, **deriv_kwargs)


class EtaUpwind(PhysUpwind):
    """Vertical upwind advection in hybrid pressure-sigma coordinates."""
    _DERIV_BWD_CLS = EtaBwdDeriv
    _DERIV_FWD_CLS = EtaFwdDeriv
    _DIM = PFULL_STR

    def __init__(self, flow, arr, pk, bk, ps, dim=None, coord=None, spacing=1,
                 order=2, fill_edge=True):
        self.flow = flow
        self.arr = arr.copy(deep=True)
        self.pk = pk
        self.bk = bk
        self.ps = ps
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge

        self.dim = dim if dim is not None else self._DIM
        self.coord = coord if coord is not None else self.arr[self._DIM]

        deriv_args = [self.pk, self.bk, self.ps]
        deriv_kwargs = dict(spacing=spacing, order=order, fill_edge=True)
        _make_derivs(self, arr.copy(deep=True), *deriv_args, **deriv_kwargs)


class LonUpwindConstP(PhysUpwind):
    """Upwind advection along a physical coordinate."""
    _DERIV_BWD_CLS = SphereEtaBwdDeriv
    _DERIV_FWD_CLS = SphereEtaFwdDeriv
    _DIM = LON_STR
    _DERIV_METHOD = 'd_dx_const_p'

    def __init__(self, flow, arr, pk, bk, ps, dim=None, coord=None, spacing=1,
                 order=2, cyclic=True, fill_edge=False):
        self.flow = flow
        self.arr = arr.copy(deep=True)
        self.pk = pk
        self.bk = bk
        self.ps = ps
        self.spacing = spacing
        self.order = order
        self.cyclic = cyclic
        self.fill_edge = fill_edge

        self.dim = dim if dim is not None else self._DIM
        self.coord = coord if coord is not None else self.arr[self._DIM]

        deriv_args = [self.pk, self.bk, self.ps]
        deriv_kwargs = dict(spacing=spacing, order=order, cyclic_lon=cyclic,
                            fill_edge_lon=fill_edge)
        _make_derivs(self, arr.copy(deep=True), *deriv_args, **deriv_kwargs)


class LatUpwindConstP(PhysUpwind):
    """Upwind advection along a physical coordinate."""
    _DERIV_BWD_CLS = SphereEtaBwdDeriv
    _DERIV_FWD_CLS = SphereEtaFwdDeriv
    _DIM = LAT_STR
    _DERIV_METHOD = 'd_dy_const_p'

    def __init__(self, flow, arr, pk, bk, ps, dim=None, coord=None, spacing=1,
                 order=2, fill_edge=True):
        self.flow = flow
        self.arr = arr.copy(deep=True)
        self.pk = pk
        self.bk = bk
        self.ps = ps
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge

        self.dim = dim if dim is not None else self._DIM
        self.coord = coord if coord is not None else self.arr[self._DIM]

        deriv_args = [self.pk, self.bk, self.ps]
        deriv_kwargs = dict(spacing=spacing, order=order,
                            fill_edge_lat=fill_edge)
        _make_derivs(self, arr.copy(deep=True), *deriv_args, **deriv_kwargs)


class SphereEtaUpwind(object):
    _X_ADVEC_CLS = LonUpwindConstP
    _Y_ADVEC_CLS = LatUpwindConstP
    _Z_ADVEC_CLS = EtaUpwind

    def __init__(self, arr, pk, bk, ps, spacing=1, order=2,
                 cyclic_lon=True, fill_edge_lon=False, fill_edge_lat=True,
                 fill_edge_vert=True):
        self.arr = arr.copy(deep=True)
        self.lat = arr[LAT_STR]
        self.pk = pk
        self.bk = bk
        self.ps = ps
        self.spacing = spacing
        self.order = order
        self.cyclic_lon = cyclic_lon
        self.fill_edge_lon = fill_edge_lon
        self.fill_edge_lat = fill_edge_lat
        self.fill_edge_vert = fill_edge_vert
        self._advec_args = [self.pk, self.bk, self.ps]

        advec_kwargs = dict(spacing=spacing, order=order,
                            fill_edge=fill_edge_lon, cyclic=cyclic_lon)
        self._advec_x_kwargs = advec_kwargs

        advec_kwargs.pop('cyclic')
        advec_kwargs.update(dict(fill_edge=fill_edge_lat))
        self._advec_y_kwargs = advec_kwargs

        advec_kwargs.update(dict(fill_edge=fill_edge_vert))
        self._advec_z_kwargs = advec_kwargs

    def advec_x_const_p(self, u):
        return self._X_ADVEC_CLS(u, self.arr.copy(deep=True),
                                 *self._advec_args,
                                 **self._advec_x_kwargs).advec()

    def advec_y_const_p(self, v):
        return self._Y_ADVEC_CLS(v, self.arr.copy(deep=True),
                                 *self._advec_args,
                                 **self._advec_y_kwargs).advec(oper='grad')

    def advec_horiz_const_p(self, u, v):
        return self.advec_x_const_p(u) + self.advec_y_const_p(v)

    def advec_z(self, omega):
        return self._Z_ADVEC_CLS(omega, self.arr.copy(deep=True),
                                 *self._advec_args,
                                 **self._advec_z_kwargs).advec()

    advec_p = advec_z

    def advec_3d(self, u, v, omega):
        return self.advec_horiz_const_p(u, v) + self.advec_p(omega)
