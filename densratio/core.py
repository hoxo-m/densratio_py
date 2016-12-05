# -*- coding: utf-8 -*-

"""
densratio.core
~~~~~~~~~~~~~~

Estimate Density Ratio p(x)/q(y)
"""

from numpy import linspace
from .uLSIF import uLSIF
from .helpers import to_numpy_matrix

def densratio(x, y, sigma_range = "auto", lambda_range = "auto",
        kernel_num = 100, verbose = True):
    """Estimate Density Ratio p(x)/q(y)

    Args:
        x: sample from p(x).
        y: sample from p(y).

    Kwargs:
        sigma_range: search range of Gaussian kernel bandwidth.
            Default "auto" means 10^-3, 10^-2, ..., 10^9.
        lambda_range: search range of regularization parameter for uLSIF.
            Default "auto" means 10^-3, 10^-2, ..., 10^9.
        kernel_num: number of kernels. Default 100.
        verbose: indicator to print messages. Default True.

    Returns:
        densratio.DensityRatio object which has `compute_density_ratio()`.

    Raises:
        ValueError: dimension of x != dimension of y

    Usage::
      >>> from scipy.stats import norm
      >>> from densratio import densratio

      >>> x = norm.rvs(size = 200, loc = 1, scale = 1./8)
      >>> y = norm.rvs(size = 200, loc = 1, scale = 1./2)
      >>> result = densratio(x, y)
      >>> print(result)

      >>> density_ratio = result.compute_density_ratio(y)
      >>> print(density_ratio)
    """

    x = to_numpy_matrix(x)
    y = to_numpy_matrix(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be same dimensions.")

    if not sigma_range or sigma_range == "auto":
        sigma_range = 10 ** linspace(-3, 1, 9)
    if not lambda_range or lambda_range == "auto":
        lambda_range = 10 ** linspace(-3, 1, 9)

    result = uLSIF(x = x, y = y,
                   sigma_range = sigma_range, lambda_range = lambda_range,
                   kernel_num = kernel_num, verbose = verbose)
    return(result)
