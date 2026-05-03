"""
densratio.core
~~~~~~~~~~~~~~

Estimate Density Ratio p(x)/q(y)
"""

from numpy import array, linspace
from .KLIEP import KLIEP as _KLIEP
from .RuLSIF import RuLSIF as _RuLSIF
from .helpers import is_numeric, to_ndarray


_METHODS = ("uLSIF", "RuLSIF", "KLIEP")


def densratio(x, y, method="uLSIF", sigma="auto", lambda_="auto", alpha=0.1,
              kernel_num=100, fold=5, verbose=True, sigma_range=None, lambda_range=None, **kwargs):
    """ Estimate Density Ratio p(x)/q(y)

    Arguments:
        x: sample from p(x).
        y: sample from q(x).
        method: "uLSIF" (default), "RuLSIF", or "KLIEP".
        sigma: search range of Gaussian kernel bandwidth.
        lambda_: search range of regularization parameter for uLSIF and RuLSIF.
        alpha: relative parameter for RuLSIF. Default 0.1.
        kernel_num: number of kernels. Default 100.
        fold: number of folds of cross validation for KLIEP. Default 5.
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
      >>> result = densratio(x, y)
      >>> print(result)

      >>> density_ratio = result.compute_density_ratio(y)
      >>> print(density_ratio)
    """

    if kwargs:
        if "lambda" in kwargs:
            lambda_ = kwargs.pop("lambda")
        if kwargs:
            raise TypeError("Unexpected keyword argument(s): {}.".format(", ".join(kwargs)))

    method = _normalize_method(method)

    if method == "uLSIF":
        return uLSIF(x, y, sigma=sigma, lambda_=lambda_, kernel_num=kernel_num, verbose=verbose,
                     sigma_range=sigma_range, lambda_range=lambda_range)

    if method == "RuLSIF":
        return RuLSIF(x, y, sigma=sigma, lambda_=lambda_, alpha=alpha, kernel_num=kernel_num, verbose=verbose,
                      sigma_range=sigma_range, lambda_range=lambda_range)

    if sigma_range is not None:
        sigma = sigma_range

    return KLIEP(x, y, sigma=sigma, kernel_num=kernel_num, fold=fold, verbose=verbose)


def uLSIF(x, y, sigma="auto", lambda_="auto", kernel_num=100, verbose=True,
          sigma_range=None, lambda_range=None, **kwargs):
    if kwargs:
        if "lambda" in kwargs:
            lambda_ = kwargs.pop("lambda")
        if kwargs:
            raise TypeError("Unexpected keyword argument(s): {}.".format(", ".join(kwargs)))

    result = _run_RuLSIF(x, y, alpha=0, sigma=sigma, lambda_=lambda_, kernel_num=kernel_num, verbose=verbose,
                         sigma_range=sigma_range, lambda_range=lambda_range)
    result.method = "uLSIF"
    return result


def RuLSIF(x, y, sigma="auto", lambda_="auto", alpha=0.1, kernel_num=100, verbose=True,
           sigma_range=None, lambda_range=None, **kwargs):
    if kwargs:
        if "lambda" in kwargs:
            lambda_ = kwargs.pop("lambda")
        if kwargs:
            raise TypeError("Unexpected keyword argument(s): {}.".format(", ".join(kwargs)))

    return _run_RuLSIF(x, y, alpha=alpha, sigma=sigma, lambda_=lambda_, kernel_num=kernel_num, verbose=verbose,
                       sigma_range=sigma_range, lambda_range=lambda_range)


def KLIEP(x, y, sigma="auto", kernel_num=100, fold=5, verbose=True):
    x = to_ndarray(x)
    y = to_ndarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be same dimensions.")

    return _KLIEP(x, y, sigma=sigma, kernel_num=kernel_num, fold=fold, verbose=verbose)


def _run_RuLSIF(x, y, alpha, sigma, lambda_, kernel_num, verbose, sigma_range=None, lambda_range=None):
    x = to_ndarray(x)
    y = to_ndarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be same dimensions.")

    if sigma_range is not None:
        sigma = sigma_range

    if lambda_range is not None:
        lambda_ = lambda_range

    sigma_range = _normalize_search_range(sigma, "sigma")
    lambda_range = _normalize_search_range(lambda_, "lambda")

    return _RuLSIF(x, y, alpha, sigma_range, lambda_range, kernel_num, verbose)


def _normalize_method(method):
    if isinstance(method, (list, tuple)):
        method = method[0]

    for candidate in _METHODS:
        if method == candidate:
            return candidate

    raise ValueError("method must be one of {}.".format(", ".join(_METHODS)))


def _normalize_search_range(value, name):
    if value is None or (isinstance(value, str) and value == "auto"):
        return 10 ** linspace(-3, 1, 9)

    if isinstance(value, str):
        raise TypeError("Invalid value for {}.".format(name))

    if is_numeric(value):
        return array([value])

    return value
