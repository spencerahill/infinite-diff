from ..deriv import CenDeriv
from . import Advec


class CenAdvec(Advec):
    """Advection using centered differencing for tracer field gradient."""
    _DERIV_CLS = CenDeriv
