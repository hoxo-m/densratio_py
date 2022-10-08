# -*- coding: utf-8 -*-

from numpy import array, matrix, ndarray, result_type


np_float = result_type(float)
try:
    import numba as nb
except ModuleNotFoundError:
    guvectorize_compute = None
else:
    _nb_float = nb.from_dtype(np_float)

    def guvectorize_compute(target: str, *, cache: bool = True):
        return nb.guvectorize([nb.void(_nb_float[:, :], _nb_float[:], _nb_float, _nb_float[:])],
                              '(m, p),(p),()->(m)',
                              nopython=True,
                              target=target,
                              cache=cache)


def is_numeric(x):
    return isinstance(x, int) or isinstance(x, float)


def to_numpy_matrix(x):
    if isinstance(x, matrix):
        return x
    elif isinstance(x, ndarray):
        if len(x.shape) == 1:
            return matrix(x).T
        else:
            return matrix(x)
    elif str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
        return x.as_matrix()
    elif not x:
        raise ValueError("Cannot transform to numpy.matrix.")
    else:
        return to_numpy_matrix(array(x))
