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
            self.coord = arr[self.dim]
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge
        self._coord_obj = self._COORD_CLS(self.coord, dim=self.dim,
                                          **coord_kwargs)
        for method in ['deriv_factor', 'deriv_prefactor']:
            setattr(self, method, getattr(self._coord_obj, method))

    def deriv(self, *args, **kwargs):
        """Derivative, incorporating physical/geometrical factors."""
        return (self._coord_obj.deriv_prefactor(*args, **kwargs) *
                self._DERIV_CLS(self.arr*self.deriv_factor(*args, **kwargs),
                                self.dim, coord=self.coord,
                                spacing=self.spacing, order=self.order,
                                fill_edge=self.fill_edge).deriv())


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


class EtaDeriv(object):
    _DERIV_CLS = FiniteDeriv
    _COORD_CLS = Eta

    def __init__(self, arr, pk, bk, ps, spacing=1, order=2, fill_edge=True,
                 **coord_kwargs):
        self.arr = arr
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
        return self._DERIV_CLS(self.arr, self.dim, coord=pfull,
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

    def __init__(self, arr, pk, bk, ps, radius=_RADEARTH):
        self.arr = arr
        self.pk = pk
        self.bk = bk
        self.ps = ps
        self.radius = radius

        self._horiz_deriv_obj = self._HORIZ_DERIV_CLS(arr)
        self._ps_horiz_deriv_obj = self._HORIZ_DERIV_CLS(ps)
        for method in ['d_dx', 'd_dy', 'horiz_grad']:
            setattr(self, method, getattr(self._horiz_deriv_obj, method))

        self._vert_deriv_obj = self._VERT_DERIV_CLS(arr, pk, bk, ps)
        for method in ['d_deta_from_pfull',
                       'd_deta_from_phalf',
                       'to_pfull_from_phalf']:
            setattr(self, method, getattr(self._vert_deriv_obj, method))

    def _horiz_deriv_const_p(self, arr, arr_deriv, ps, ps_deriv):
        """Horizontal derivative in single direction at constant pressure."""
        darr_deta = self.d_deta_from_pfull(arr)
        bk_at_pfull = self.to_pfull_from_phalf(self.bk)
        da_deta = self.d_deta_from_phalf(self.pk)
        db_deta = self.d_deta_from_phalf(self.bk)

        return arr_deriv + (darr_deta * bk_at_pfull * ps_deriv /
                            (da_deta + db_deta*ps))

    def d_dx_const_p(self):
        return self._horiz_deriv_const_p(
            self.arr, self.d_dx(), self.ps,
            self._ps_horiz_deriv_obj.d_dx()
        )

    def d_dy_const_p(self):
        return self._horiz_deriv_const_p(
            self.arr, self.d_dy(), self.ps,
            self._ps_horiz_deriv_obj.d_dy()
        )

    def horiz_grad_const_p(self):
        return self.d_dx_const_p() + self.d_dy_const_p()


class SphereEtaFwdDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereFwdDeriv
    _VERT_DERIV_CLS = EtaFwdDeriv


class SphereEtaBwdDeriv(SphereEtaDeriv):
    """Derivatives on the sphere with hybrid sigma-pressure in the vertical."""
    _HORIZ_DERIV_CLS = SphereBwdDeriv
    _VERT_DERIV_CLS = EtaBwdDeriv
