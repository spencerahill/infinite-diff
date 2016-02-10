from .. import CenDeriv
from . import Advec


class CenAdvec(Advec):
    """Advection using centered differencing for tracer field gradient."""
    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True):
        super(CenAdvec, self).__init__(flow, arr, dim, coord=coord,
                                       spacing=spacing, order=order,
                                       fill_edge=fill_edge)
        self._deriv_obj = CenDeriv(arr, dim, coord=coord)
