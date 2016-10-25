# -*- coding: utf-8 -*-

from uLSIF import uLSIF

def densratio(x, y):
    """
    Estimate Density Ratio p(x)/q(y)
    """
    result = uLSIF(x, y)
    return(result)
