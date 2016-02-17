"""Physical coordinates."""


class Coord(object):
    """Generic base class for physical coordinates."""
    _POSSIBLY_CYCLIC = True

    def _prep_dim(self, dim):
        msg = ("Specified dim '{}' does not match any dim in self.arr.  "
               "self.arr.dims: {}").format(dim, self.arr.dims)
        if dim in self.arr.dims:
            return dim
        if dim is None:
            if self.arr.shape == 1:
                return self.arr.dims[0]
        raise ValueError(msg)

    def __init__(self, arr, dim=None, cyclic=False):
        self.arr = arr
        self.dim = self._prep_dim(dim)
        if cyclic and not self._POSSIBLY_CYCLIC:
            raise ValueError("The coordinate cannot be cyclic in a physically "
                             "meaningful way.")
        self.cyclic = cyclic

    def __getitem__(self, key):
        return self.arr[key]

    def deriv_prefactor(self, *args, **kwargs):
        raise NotImplementedError

    def deriv_factor(self, *args, **kwargs):
        raise NotImplementedError
