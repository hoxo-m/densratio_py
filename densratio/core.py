# -*- coding: utf-8 -*-

"""
densratio.core
~~~~~~~~~~~~~~

Estimate Density Ratio p(x)/q(y)
"""

from numpy import linspace
from .RuLSIF import RuLSIF
from .helpers import to_numpy_matrix


def densratio(x, y, alpha=0, sigma_range="auto", lambda_range="auto", kernel_num=100, verbose=True):
    """ Estimate alpha-mixture Density Ratio p(x)/(alpha*p(x) + (1 - alpha)*q(x))

    Arguments:
        x: sample from p(x).
        y: sample from q(x).
        alpha: Default 0 - corresponds to ordinary density ratio.
        sigma_range: search range of Gaussian kernel bandwidth.
            Default "auto" means 10^-3, 10^-2, ..., 10^9.
        lambda_range: search range of regularization parameter for uLSIF.
            Default "auto" means 10^-3, 10^-2, ..., 10^9.
        kernel_num: number of kernels. Default 100.
        verbose: indicator to print messages. Default True.

    Returns:
        densratio.DensityRatio object which has `compute_density_ratio()`.

    Raises:
        ValueError: if dimension of x != dimension of y

    Usage::
      >>> from scipy.stats import norm
      >>> from densratio import densratio

      >>> x = norm.rvs(size=200, loc=1, scale=1./8)
      >>> y = norm.rvs(size=200, loc=1, scale=1./2)
      >>> result = densratio(x, y, alpha=0.7)
      >>> print(result)

      >>> density_ratio = result.compute_density_ratio(y)
      >>> print(density_ratio)
    """

    x = to_numpy_matrix(x)
    y = to_numpy_matrix(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be same dimensions.")

    if isinstance(sigma_range, str) and sigma_range != "auto":
        raise TypeError("Invalid value for sigma_range.")

    if isinstance(lambda_range, str) and lambda_range != "auto":
        raise TypeError("Invalid value for lambda_range.")

    if sigma_range is None or (isinstance(sigma_range, str) and sigma_range == "auto"):
        sigma_range = 10 ** linspace(-3, 9, 13)

    if lambda_range is None or (isinstance(lambda_range, str) and lambda_range == "auto"):
        lambda_range = 10 ** linspace(-3, 9, 13)

    result = RuLSIF(x, y, alpha, sigma_range, lambda_range, kernel_num, verbose)
    return result
