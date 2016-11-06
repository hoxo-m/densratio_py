# -*- coding: utf-8 -*-

"""
densratio.core
~~~~~~~~~~~~~~

Estimate Density Ratio p(x)/q(y)
"""

from .uLSIF import uLSIF

def densratio(x, y, sigma = "auto", lambda_ = "auto",
        kernel_num = 100, verbose = True):
    """
    Estimate Density Ratio p(x)/q(y)

    :param x: sample from p(x)
    :param y: sample from p(y)
    :param sigma: search range of Gaussian kernel bandwidth
    :param lambda_: search range of regularization parameter for uLSIF
    :param kernel_num: number of kernels
    :param verbose: indicator to print messages
    :return: :class:`DensityRatio <DensityRatio>` object
    :rtype: densratio.DensityRatio

    Usage::
      >>> from scipy.stats import norm
      >>> from densratio import densratio

      >>> x = norm.rvs(size = 200, loc = 1, scale = 1./8)
      >>> y = norm.rvs(size = 200, loc = 1, scale = 1./2)
      >>> result = densratio(x, y)
      >>> print(result)
      Method: uLSIF

      Kernel Information:
        Kernel type: Gaussian RBF
        Number of kernels: 100
        Bandwidth(sigma): 0.1
        Centers: array([ 1.04719547, 0.98294441, 1.06190142, 1.14145367, 1.18276349,..

      Kernel Weights(alpha):
        array([ 0.25340775, 0.40951303, 0.27002649, 0.17998821, 0.14524305,..

      Regularization Parameter(lambda): 0.1

      The Function to Estimate Density Ratio:
        compute_density_ratio(x)
    """
    result = uLSIF(x = x, y = y, sigma_range = sigma, lambda_range = lambda_,
                   kernel_num = kernel_num, verbose = verbose)
    return(result)
