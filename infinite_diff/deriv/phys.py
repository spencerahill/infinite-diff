from .._constants import LON_STR, LAT_STR, PFULL_STR, _RADEARTH
from ..coord import Coord, Lon, Lat, Eta
from . import FiniteDeriv, FwdDeriv, BwdDeriv, CenDeriv


class PhysDeriv(object):
    """Derivatives in physical space."""
    _COORD_CLS = Coord
    _DERIV_CLS = FiniteDeriv

    def __init__(self, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True, **coord_kwargs):
        self.arr = arr
        self.dim = dim
        if coord is None:
            coord = arr[self.dim]
        self._deriv_kwargs = dict(spacing=spacing, order=order,
                                  fill_edge=fill_edge)
        self._coord_obj = self._COORD_CLS(coord, dim=self.dim, **coord_kwargs)
        self._deriv_obj = self._DERIV_CLS(arr, self.dim, coord=coord)

    def _deriv(self, *args, **kwargs):
        return self._DERIV_CLS(
            self.arr*self._coord_obj.deriv_factor(*args, **kwargs),
            self.dim
        ).deriv(**self._deriv_kwargs)

    def deriv(self, *args, **kwargs):
        """Derivative, incorporating physical/geometrical factors."""
        return (self._coord_obj.deriv_prefactor(*args, **kwargs) *
                self._deriv(*args))


class LonDeriv(PhysDeriv):
    _COORD_CLS = Lon


class LonFwdDeriv(LonDeriv):
    _DERIV_CLS = FwdDeriv


class LonBwdDeriv(LonDeriv):
    _DERIV_CLS = BwdDeriv


class LonCenDeriv(LonDeriv):
    _DERIV_CLS = CenDeriv


class LatDeriv(PhysDeriv):
    _COORD_CLS = Lat


class LatFwdDeriv(LatDeriv):
    _DERIV_CLS = FwdDeriv


class LatBwdDeriv(LatDeriv):
    _DERIV_CLS = BwdDeriv


class LatCenDeriv(LatDeriv):
    _DERIV_CLS = CenDeriv


class EtaDeriv(PhysDeriv):
    _COORD_CLS = Eta

    def __init__(self, arr, pk, bk, ps, spacing=1, order=2, fill_edge=True,
                 **coord_kwargs):
        self.arr = arr
        self.dim = PFULL_STR
        self.ps = ps

        self._coord_obj = self._COORD_CLS(pk, bk, self.arr[self.dim],
                                          **coord_kwargs)
        self.pfull = self._coord_obj.pfull

        self._deriv_kwargs = dict(spacing=spacing, order=order,
                                  fill_edge=fill_edge)
        self._deriv_obj = self._DERIV_CLS(arr, self.dim)
        for method in ['phalf_from_ps',
                       'to_pfull_from_phalf',
                       'pfull_from_ps',
                       'd_deta_from_phalf',
                       'd_deta_from_pfull',
                       'dp_from_ps']:
            setattr(self, method, getattr(self._coord_obj, method))

    def _deriv(self):
        pfull = self.pfull_from_ps(self.ps)
        return self._DERIV_CLS(self.arr, self.dim,
                               coord=pfull).deriv(**self._deriv_kwargs)

    deriv = _deriv


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
        self._x_deriv_obj = self._X_DERIV_CLS(arr, x_dim, coord=x_coord,
                                              **kwargs)
        self._y_deriv_obj = self._Y_DERIV_CLS(arr, y_dim, coord=y_coord,
                                              **kwargs)

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

    def __init__(self, arr, x_coord=None, y_coord=None, **kwargs):
        self.arr = arr
        self._x_deriv_obj = self._X_DERIV_CLS(arr, LON_STR, coord=x_coord,
                                              **kwargs)
        self._y_deriv_obj = self._Y_DERIV_CLS(arr, LAT_STR, coord=y_coord,
                                              **kwargs)

    def d_dx(self):
        return self._x_deriv_obj.deriv(self.arr[LAT_STR])

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


class SphereEtaDeriv(object):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereDeriv
    _VERT_DERIV_CLS = EtaDeriv

    def __init__(self, arr, pk, bk, ps, radius=_RADEARTH, **kwargs):
        self.arr = arr
        self._x = arr[LON_STR]
        self._y = arr[LAT_STR]
        self._z = arr[PFULL_STR]

        self._horiz_deriv_obj = self._HORIZ_DERIV_CLS(arr)
        self.d_dx = self._horiz_deriv_obj.d_dx
        self.d_dy = self._horiz_deriv_obj.d_dy

        self._vert_deriv_obj = self._VERT_DERIV_CLS(arr, pk, bk, ps)
        self.d_deta_from_pfull = self._vert_deriv_obj.d_deta_from_pfull
        self.d_deta_from_phalf = self._vert_deriv_obj.d_deta_from_phalf
        self.to_pfull_from_phalf = self._vert_deriv_obj.to_pfull_from_phalf

    def _horiz_deriv_const_p(self, arr, arr_deriv, ps, ps_deriv):
        """Horizontal derivative in single direction at constant pressure."""
        darr_deta = self.d_deta_from_pfull(arr)
        bk_at_pfull = self.to_pfull_from_phalf(self.bk)
        da_deta = self.d_deta_from_phalf(self.pk)
        db_deta = self.d_deta_from_phalf(self.bk)

        return arr_deriv + (darr_deta * bk_at_pfull * ps_deriv /
                            (da_deta + db_deta*ps))

    def d_dx_const_p(self):
        return self._horiz_deriv_const_p(self.arr, self.d_dx(self.arr),
                                         self.ps, self.d_dx(self.ps))

    def d_dy_const_p(self):
        return self._horiz_deriv_const_p(self.arr, self.d_dy(self.arr),
                                         self.ps, self.d_dy(self.ps))

    def horiz_grad_const_p(self):
        return self.d_dx_const_p(self.arr) + self.d_dy_const_p(self.arr)


class SphereEtaFwdDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereFwdDeriv
    _VERT_DERIV_CLS = EtaFwdDeriv


class SphereEtaBwdDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereBwdDeriv
    _VERT_DERIV_CLS = EtaBwdDeriv
