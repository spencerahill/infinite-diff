"""Upwind advection.

Upwind advection uses one-sided differencing in the upstream direction of the
flow to compute the tracer field derivative.  See
https://en.wikipedia.org/wiki/Upwind_scheme for formulae of upwind schemes of
first, second, and third order accuracy.
"""
from ..deriv import FwdDeriv, BwdDeriv
from . import Advec


class Upwind(Advec):
    """Upwind advection."""
    _DERIV_BWD_CLS = BwdDeriv
    _DERIV_FWD_CLS = FwdDeriv
    _DERIV_METHOD = 'deriv'

    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True):
        super(Upwind, self).__init__(flow, arr, dim, coord=coord,
                                     spacing=spacing, order=order,
                                     fill_edge=fill_edge)
        self._deriv_bwd_obj = self._DERIV_BWD_CLS(
            self.arr, self.dim, coord=self.coord, spacing=self.spacing,
            order=self.order, fill_edge=True
        )
        self._deriv_fwd_obj = self._DERIV_FWD_CLS(
            self.arr, self.dim, coord=self.coord, spacing=self.spacing,
            order=self.order, fill_edge=True
        )
        self._deriv_bwd = getattr(self._deriv_bwd_obj, self._DERIV_METHOD)
        self._deriv_fwd = getattr(self._deriv_fwd_obj, self._DERIV_METHOD)

    def _flow_neg_pos(self, reverse_dim=False):
        """Create negative- and positive-only arrays for upwind advection.

        :out: flow_neg, flow_pos xarray.DataArrays with shape and coords
            identical to `flow1, but with, respectively, all positive and
            negative values set to 0 (or the reverse if `reverse_dim` is
            `True`).
        """
        flow_neg = self.flow.copy()
        flow_neg.values[self.flow.values >= 0] = 0.
        flow_pos = self.flow.copy()
        flow_pos.values[self.flow.values < 0] = 0.
        if not reverse_dim:
            return flow_neg, flow_pos
        return flow_pos, flow_neg

    def _swap_bwd_fwd_edges(self, bwd, fwd):
        """Forward diff on left edge; backward diff on right edge."""
        edge_left = {self.dim: 0}
        edge_right = {self.dim: -1}
        bwd[edge_left] = fwd[edge_left]
        fwd[edge_right] = bwd[edge_right]
        return bwd, fwd

    def _derivs_bwd_fwd(self):
        """Generate forward and backward differencing derivs for upwind.

        Order of accuracy decreases moving towards edge (right edge for
        forward, left edge for backward) as the differencing stencil starts to
        extend over the domain edge.  At the edge itself, the opposite signed
        differencing is used with the same order of accuracy as in the
        interior.
        """
        bwd = self._deriv_bwd()
        fwd = self._deriv_fwd()
        # Forward diff on left edge; backward diff on right edge.
        return self._swap_bwd_fwd_edges(bwd, fwd)

    def advec(self):
        """
        Upwind differencing scheme for advection.

        In interior, forward differencing for negative flow, and backward
        differencing for positive flow.

        :param arr: Field being advected.
        :param flow: Flow that is advecting the field.
        """
        bwd, fwd = self._derivs_bwd_fwd()
        neg, pos = self._flow_neg_pos()
        advec_arr = pos*bwd + neg*fwd
        if not self.fill_edge:
            slice_middle = {self.dim: slice(self.order, -self.order)}
            advec_arr = advec_arr[slice_middle]
        return advec_arr
