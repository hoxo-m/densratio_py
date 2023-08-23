from numpy import array, ndarray, result_type


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


def to_ndarray(x):
    if isinstance(x, ndarray):
        if len(x.shape) == 1:
            return x.reshape(-1, 1)
        else:
            return x
    elif str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
        return x.values
    elif not x:
        raise ValueError("Cannot transform to numpy.ndarray.")
    else:
        return to_ndarray(array(x))
