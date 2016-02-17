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
from . import phys
from .phys import PhysDeriv, LonDeriv, LatDeriv, EtaDeriv, SphereDeriv
from .phys import LonFwdDeriv, LonBwdDeriv, LonCenDeriv
from .phys import LatFwdDeriv, LatBwdDeriv, LatCenDeriv
from .phys import EtaFwdDeriv, EtaBwdDeriv, EtaCenDeriv
from .phys import SphereFwdDeriv, SphereBwdDeriv
from .phys import SphereEtaDeriv, SphereEtaBwdDeriv, SphereEtaFwdDeriv
