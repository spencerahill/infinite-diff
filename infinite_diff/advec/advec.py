from .. import FiniteDeriv


class Advec(object):
    """Base class for advection."""
    def _arr_gradient(self):
        """Compute the gradient of the field."""
        return self._deriv_obj.deriv(spacing=self.spacing, order=self.order,
                                     fill_edge=self.fill_edge)

    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True):
        self.flow = flow
        self.arr = arr
        self.dim = dim
        self.coord = coord
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge
        self._deriv_obj = FiniteDeriv(arr, dim, coord=coord)

    def advec(self):
        """Advect the tracer array with the flow."""
        return self.flow * self._arr_gradient()
