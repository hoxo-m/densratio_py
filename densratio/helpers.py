# -*- coding: utf-8 -*-

from numpy import array, matrix, ndarray

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
    elif not x:
        raise ValueError("Cannot transform to numpy.matrix.")
    else:
        return to_numpy_matrix(array(x))
