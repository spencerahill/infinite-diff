import warnings

from .._constants import LON_STR, LAT_STR, PFULL_STR, _RADEARTH
from ..utils import to_radians, wraparound
from ..coord import Coord, Lon, Lat, Eta
from . import FiniteDeriv, FwdDeriv, BwdDeriv, CenDeriv


class PhysDeriv(object):
    """Derivatives in physical space."""
    _COORD_CLS = Coord
    _DERIV_CLS = FiniteDeriv
    _WRAP_CIRCUMF = 0
    _WRAP_LEFT_TO_RIGHT = 0
    _WRAP_RIGHT_TO_LEFT = 0

    def _get_coord(self, coord):
        if coord is None:
            return self.arr[self.dim].copy(deep=True)
        return coord

    def __init__(self, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True, **coord_kwargs):
        self.arr = arr.copy(deep=True)
        self.dim = dim
        self.coord = self._get_coord(coord)
        self._orig_coord_values = self.coord.values
        self.spacing = spacing
        self.order = order
        self.cyclic = coord_kwargs.get('cyclic', False)

        self._coord_obj = self._COORD_CLS(self.coord, dim=self.dim,
                                          **coord_kwargs)
        for attr in ['deriv_factor', 'deriv_prefactor']:
            setattr(self, attr, getattr(self._coord_obj, attr))

        if fill_edge and self.cyclic:
            warnings.warn("Overriding 'fill_edge' value of True, because "
                          "coord is cyclic")
        self.fill_edge = False if self.cyclic else fill_edge

    def _wrap(self, arr):
        if self.cyclic:
            return wraparound(
                arr.copy(deep=True), self.dim,
                left_to_right=self._WRAP_LEFT_TO_RIGHT*self.order,
                right_to_left=self._WRAP_RIGHT_TO_LEFT*self.order,
                circumf=self._WRAP_CIRCUMF, spacing=self.spacing
            )
        return arr.copy(deep=True)

    def _prep_coord(self, coord):
        return coord

    def deriv(self, *args, **kwargs):
        """Derivative, incorporating physical/geometrical factors."""
        arr = self._wrap(self.arr.copy(deep=True) *
                         self.deriv_factor(*args, **kwargs))
        coord = self._prep_coord(arr[self.dim].copy(deep=True))
        darr = (self._DERIV_CLS(arr.copy(deep=True), self.dim,
                                coord=coord.copy(deep=True),
                                spacing=self.spacing, order=self.order,
                                fill_edge=self.fill_edge).deriv() *
                self._coord_obj.deriv_prefactor(*args, **kwargs))
        return darr


class LonDeriv(PhysDeriv):
    _COORD_CLS = Lon
    _WRAP_CIRCUMF = 360.

    def _prep_coord(self, coord):
        return to_radians(coord.copy(deep=True))


class LonFwdDeriv(LonDeriv):
    _DERIV_CLS = FwdDeriv
    _WRAP_LEFT_TO_RIGHT = 1


class LonBwdDeriv(LonDeriv):
    _DERIV_CLS = BwdDeriv
    _WRAP_RIGHT_TO_LEFT = 1


class LonCenDeriv(LonDeriv):
    _DERIV_CLS = CenDeriv
    _WRAP_LEFT_TO_RIGHT = 1
    _WRAP_RIGHT_TO_LEFT = 1


class LatDeriv(PhysDeriv):
    _COORD_CLS = Lat

    def _prep_coord(self, coord):
        return to_radians(coord.copy(deep=True))


class LatFwdDeriv(LatDeriv):
    _DERIV_CLS = FwdDeriv


class LatBwdDeriv(LatDeriv):
    _DERIV_CLS = BwdDeriv


class LatCenDeriv(LatDeriv):
    _DERIV_CLS = CenDeriv


class EtaDeriv(object):
    _DERIV_CLS = FiniteDeriv
    _COORD_CLS = Eta

    def __init__(self, arr, pk, bk, ps, spacing=1, order=2, fill_edge=True,
                 **coord_kwargs):
        self.arr = arr.copy(deep=True)
        self.dim = PFULL_STR
        self.ps = ps
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge

        self._coord_obj = self._COORD_CLS(pk, bk, self.arr[self.dim],
                                          **coord_kwargs)
        self.pfull = self._coord_obj.pfull
        for method in ['phalf_from_ps',
                       'to_pfull_from_phalf',
                       'pfull_from_ps',
                       'd_deta_from_phalf',
                       'd_deta_from_pfull',
                       'dp_from_ps']:
            setattr(self, method, getattr(self._coord_obj, method))

    def deriv(self):
        pfull = self.pfull_from_ps(self.ps)
        return self._DERIV_CLS(self.arr.copy(deep=True), self.dim, coord=pfull,
                               spacing=self.spacing, order=self.order,
                               fill_edge=self.fill_edge).deriv()


class EtaFwdDeriv(EtaDeriv):
    _DERIV_CLS = FwdDeriv


class EtaBwdDeriv(EtaDeriv):
    _DERIV_CLS = BwdDeriv


class EtaCenDeriv(EtaDeriv):
    _DERIV_CLS = CenDeriv


class HorizPhysDeriv(object):
    """Horizontal derivatives."""
    _X_DERIV_CLS = PhysDeriv
    _Y_DERIV_CLS = PhysDeriv

    def __init__(self, arr, x_dim, y_dim, x_coord=None, y_coord=None,
                 **kwargs):
        self._x_deriv_obj = self._X_DERIV_CLS(arr.copy(deep=True), x_dim,
                                              coord=x_coord, **kwargs)
        self._y_deriv_obj = self._Y_DERIV_CLS(arr.copy(deep=True), y_dim,
                                              coord=y_coord, **kwargs)

    def d_dx(self, *args, **kwargs):
        return self._x_deriv_obj.deriv(*args, **kwargs)

    def d_dy(self, *args, **kwargs):
        return self._y_deriv_obj.deriv(*args, **kwargs)

    def horiz_grad(self, *args, **kwargs):
        return self.d_dx(*args, **kwargs) + self.d_dy(*args, **kwargs)


class SphereDeriv(HorizPhysDeriv):
    """Derivatives for data on a sphere."""
    _X_DERIV_CLS = LonDeriv
    _Y_DERIV_CLS = LatDeriv

    def __init__(self, arr, x_coord=None, y_coord=None, cyclic_lon=True,
                 fill_edge_lon=False, fill_edge_lat=True, **kwargs):
        self.arr = arr.copy(deep=True)
        self.cyclic_lon = cyclic_lon
        self._x_deriv_obj = self._X_DERIV_CLS(
            arr.copy(deep=True), LON_STR, coord=x_coord, cyclic=cyclic_lon,
            fill_edge=fill_edge_lon, **kwargs
        )
        self._y_deriv_obj = self._Y_DERIV_CLS(
            arr.copy(deep=True), LAT_STR, coord=y_coord,
            fill_edge=fill_edge_lat, **kwargs
        )
        self.d_dy = self._y_deriv_obj.deriv

    def d_dx(self):
        return self._x_deriv_obj.deriv(self.arr.copy(deep=True))

    def horiz_grad(self):
        return self.d_dx() + self.d_dy(oper='grad')


class SphereFwdDeriv(SphereDeriv):
    """Derivatives for data on a sphere."""
    _X_DERIV_CLS = LonFwdDeriv
    _Y_DERIV_CLS = LatFwdDeriv


class SphereBwdDeriv(SphereDeriv):
    """Derivatives for data on a sphere."""
    _X_DERIV_CLS = LonBwdDeriv
    _Y_DERIV_CLS = LatBwdDeriv


class SphereCenDeriv(SphereDeriv):
    """Derivatives for data on a sphere."""
    _X_DERIV_CLS = LonCenDeriv
    _Y_DERIV_CLS = LatCenDeriv


class SphereEtaDeriv(object):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereDeriv
    _VERT_DERIV_CLS = EtaDeriv

    def __init__(self, arr, pk, bk, ps, spacing=1, order=2, cyclic_lon=True,
                 fill_edge_lon=False, fill_edge_lat=True, fill_edge_vert=True,
                 radius=_RADEARTH):
        self.arr = arr.copy(deep=True)
        self.pk = pk
        self.bk = bk
        self.ps = ps
        self.spacing = spacing
        self.order = order
        self.cyclic_lon = cyclic_lon
        self.fill_edge_lon = fill_edge_lon
        self.fill_edge_lat = fill_edge_lat
        self.fill_edge_vert = fill_edge_vert
        self.radius = radius

        horiz_deriv_kwargs = dict(
            spacing=spacing, order=order, cyclic_lon=cyclic_lon,
            fill_edge_lon=fill_edge_lon, fill_edge_lat=fill_edge_lat,
            radius=radius
        )
        self._horiz_deriv_obj = self._HORIZ_DERIV_CLS(arr.copy(deep=True),
                                                      **horiz_deriv_kwargs)
        self._ps_horiz_deriv_obj = self._HORIZ_DERIV_CLS(ps.copy(deep=True),
                                                         **horiz_deriv_kwargs)
        for method in ['d_dx', 'd_dy', 'horiz_grad']:
            setattr(self, method, getattr(self._horiz_deriv_obj, method))

        vert_deriv_kwargs = dict(spacing=spacing, order=order,
                                 fill_edge=fill_edge_vert)
        self._vert_deriv_obj = self._VERT_DERIV_CLS(
            arr.copy(deep=True), pk, bk, ps, **vert_deriv_kwargs
        )
        for method in ['d_deta_from_pfull',
                       'd_deta_from_phalf',
                       'to_pfull_from_phalf']:
            setattr(self, method, getattr(self._vert_deriv_obj, method))
        self.d_dp = self._vert_deriv_obj.deriv

    def _horiz_deriv_const_p(self, arr, arr_deriv, ps, ps_deriv):
        """Horizontal derivative in single direction at constant pressure."""
        darr_deta = self.d_deta_from_pfull(arr.copy(deep=True))
        bk_at_pfull = self.to_pfull_from_phalf(self.bk)
        da_deta = self.d_deta_from_phalf(self.pk)
        db_deta = self.d_deta_from_phalf(self.bk)
        return arr_deriv + (darr_deta * bk_at_pfull * ps_deriv /
                            (da_deta + db_deta*ps))

    def d_dx_const_p(self):
        return self._horiz_deriv_const_p(
            self.arr.copy(deep=True), self.d_dx(), self.ps,
            self._ps_horiz_deriv_obj.d_dx()
        )

    def d_dy_const_p(self, oper='grad'):
        return self._horiz_deriv_const_p(
            self.arr.copy(deep=True), self.d_dy(oper=oper), self.ps,
            self._ps_horiz_deriv_obj.d_dy(oper=oper)
        )

    def horiz_grad_const_p(self):
        return self.d_dx_const_p() + self.d_dy_const_p(oper='grad')

    def grad_3d(self):
        return self.horiz_grad_const_p(oper='grad') + self.d_dp()


class SphereEtaFwdDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereFwdDeriv
    _VERT_DERIV_CLS = EtaFwdDeriv


class SphereEtaBwdDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereBwdDeriv
    _VERT_DERIV_CLS = EtaBwdDeriv


class SphereEtaCenDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereCenDeriv
    _VERT_DERIV_CLS = EtaCenDeriv
