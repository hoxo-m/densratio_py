import numpy as np

from numpy import array, ndarray, result_type
from warnings import warn


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


def alpha_normalize(values: ndarray, alpha: float) -> ndarray:
    """
    Normalizes values less than 1 so the minimum value to replace 0 is symmetrical to alpha^-1
    with respect to the natural logarithm.

    Arguments:
        values (numpy.array): A vector to normalize.
        alpha (float): The nonnegative normalization term.

    Returns:
        Normalized numpy.array object that preserves the order and the number of unique input argument values.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError('\'values\' must a 1d vector.')

    if alpha <= 0.:
        if alpha < 0.:
            warn('\'alpha\' is negative, normalization aborted.', RuntimeWarning)

        return values

    a = 1. - alpha
    inserted = last_value = 1.
    outcome = np.empty(values.shape, dtype=values.dtype)

    values_argsort = np.argsort(values)
    for i in np.flip(values_argsort):
        value = values[i]
        if value >= 1.:
            outcome[i] = value
            continue

        if value < last_value:
            new_value = inserted - a * (last_value - value)
            inserted = np.nextafter(inserted, 0.) if new_value == inserted else new_value
            last_value = value
        else:
            assert value == last_value

        outcome[i] = inserted

    if inserted <= 0.:
        warn(f'Normalized vector contains at least one nonpositive [min={inserted}] value.', RuntimeWarning)
    elif inserted < alpha:
        warn(f'Normalized vector contains at least one value [min={inserted}] less than alpha [{alpha}].',
             RuntimeWarning)

    return outcome
