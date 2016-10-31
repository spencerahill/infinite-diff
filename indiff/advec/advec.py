from ..deriv import FiniteDeriv


class Advec(object):
    """Base class for advection."""
    _DERIV_CLS = FiniteDeriv

    def _arr_gradient(self):
        """Compute the gradient of the field."""
        return self._deriv_obj.deriv()

    def __init__(self, flow, arr, dim, coord=None, spacing=1, order=2,
                 fill_edge=True):
        self.flow = flow
        self.arr = arr
        self.dim = dim
        self.coord = coord
        self.spacing = spacing
        self.order = order
        self.fill_edge = fill_edge
        self._deriv_obj = self._DERIV_CLS(self.arr, self.dim, coord=self.coord,
                                          spacing=self.spacing,
                                          order=self.order,
                                          fill_edge=self.fill_edge)

    def advec(self):
        """Advect the tracer array with the flow."""
        return self.flow * self._arr_gradient()
