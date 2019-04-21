# -*- coding: utf-8 -*-

from pprint import pformat
from re import sub


class DensityRatio:
    """Density Ratio."""
    def __init__(self, method, alpha, theta, lambda_, alpha_PE, alpha_KL, kernel_info, compute_density_ratio):
        self.method = method
        self.alpha = alpha
        self.theta = theta
        self.lambda_ = lambda_
        self.alpha_PE = alpha_PE
        self.alpha_KL = alpha_KL
        self.kernel_info = kernel_info
        self.compute_density_ratio = compute_density_ratio

    def __str__(self):
        return """
Method: %(method)s

Alpha: %(alpha)s

Kernel Information:
%(kernel_info)s

Kernel Weights (theta):
  %(theta)s

Regularization Parameter (lambda): %(lambda_)s

Alpha-Relative PE-Divergence: %(alpha_PE)s

Alpha-Relative KL-Divergence: %(alpha_KL)s

Function to Estimate Density Ratio:
  compute_density_ratio(x)
  
"""[1:-1] % dict(method=self.method, kernel_info=self.kernel_info, alpha=self.alpha, theta=my_format(self.theta), lambda_=self.lambda_, alpha_PE=self.alpha_PE, alpha_KL=self.alpha_KL)


class KernelInfo:
    """Kernel Information."""
    def __init__(self, kernel_type, kernel_num, sigma, centers):
        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.sigma = sigma
        self.centers = centers

    def __str__(self):
        return """
  Kernel type: %(kernel_type)s
  Number of kernels: %(kernel_num)s
  Bandwidth(sigma): %(sigma)s
  Centers: %(centers)s
"""[1:-1] % dict(kernel_type=self.kernel_type, kernel_num=self.kernel_num, sigma=self.sigma, centers=my_format(self.centers))


def my_format(str):
    return sub(r"\s+" , " ", (pformat(str).split("\n")[0] + ".."))
