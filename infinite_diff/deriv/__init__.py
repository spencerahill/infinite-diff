"""Finite-differencing-based numerical derivative methods.

See https://en.wikipedia.org/wiki/Finite_difference_coefficient for formulae
for forward, backward, and centered differencing stencils of various orders.
"""
from . import finite
from .finite import FiniteDeriv
from . import one_sided
from .one_sided import OneSidedDeriv, FwdDeriv, BwdDeriv
from . import centered
from .centered import CenDeriv
