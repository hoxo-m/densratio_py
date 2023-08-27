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


def semi_stratified_sample(data: ndarray, size: int) -> ndarray:
    if data.ndim > 2:
        raise ValueError('Only single and 2d arrays are supported.')
    if not size:
        return np.empty(0)

    data_length = data.shape[0]
    result = np.arange(data_length, dtype=int)
    if size == data_length:
        np.random.shuffle(result)
        return result
    if size < 0:
        raise ValueError('Sample size must be a non-negative integer number.')
    if size > data_length:
        raise ValueError('Sample size cannot exceed the shape of input data.')

    dims = data.shape[1]
    indexed = np.column_stack((data, result))
    result = np.empty(0, dtype=indexed.dtype)

    samples_no = size // dims
    if samples_no:
        percentiles = np.linspace(0., 100., num=samples_no, endpoint=False)[1:]

        for d in range(dims):
            column = indexed[..., d]
            quantiles = np.append(column.min(), np.percentile(column, percentiles))
            indices = []
            i, sample_size = 0, 1

            while i < samples_no:
                left = quantiles[i]
                i += 1
                right = np.Inf if i == samples_no else quantiles[i]
                try:
                    indices.extend(np.random.choice(
                        indexed[(left <= column) & (column < right), dims],
                        size=sample_size,
                        replace=False))
                except ValueError:
                    sample_size += 1
                    continue
                else:
                    sample_size = 1

            indexed = indexed[~np.isin(indexed[:, dims], indices)]
            result = np.append(result, indices)

    result = np.append(
        result,
        np.random.choice(indexed[..., dims], size=size - result.size, replace=False)).astype(int)
    np.random.shuffle(result)
    return result
