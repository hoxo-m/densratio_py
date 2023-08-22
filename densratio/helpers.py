import numpy as np

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


def to_ndarray(x):
    if isinstance(x, ndarray):
        if len(x.shape) == 1:
            return x.reshape(-1, 1)
        else:
            return x
    elif str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
        return x.values
    elif not x:
        raise ValueError("Cannot transform to numpy.matrix.")
    else:
        return to_ndarray(array(x))


def semi_stratified_sample(data: ndarray, samples: int) -> ndarray:
    ndims = data.ndim
    if ndims > 2:
        raise ValueError('Only single and 2d arrays are supported.')
    if not samples:
        return np.empty(0)

    data_length = data.shape[0]
    result = np.arange(data_length, dtype=int)
    if samples == data_length:
        np.random.shuffle(result)
        return result
    if samples < 0:
        raise ValueError('Number of samples must be a non-negative integer number.')
    if samples > data_length:
        raise ValueError('Number of samples cannot exceed the shape of input data.')

    dims = data.shape[1] if 2 == ndims else 1
    indexed = np.column_stack((data, result))
    result = np.empty(0, dtype=indexed.dtype)

    samples_no = samples // dims
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
        np.random.choice(indexed[..., dims], size=samples-result.size, replace=False)).astype(int)
    np.random.shuffle(result)
    return result
