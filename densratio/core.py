# -*- coding: utf-8 -*-

from uLSIF import uLSIF

def densratio(x, y, sigma_range = None, lambda_range = None,
        kernel_num = 100, verbose = True):
    """
    Estimate Density Ratio p(x)/q(y)
    """
    result = uLSIF(x = x, y = y, sigma_range = sigma_range,
                   lambda_range = lambda_range,
                   kernel_num = kernel_num, verbose = verbose)
    return(result)
