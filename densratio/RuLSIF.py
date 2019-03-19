"""
Relative Unconstrained Least-Squares Fitting (RuLSIF)

A Python Implementation of RuLSIF for densratio.py
Author: Ameya Daigavane
Reference: 'Change-point detection in time-series data by relative density-ratio estimation
            Song Liu, Makoto Yamada, Nigel Collier and Masashi Sugiyama,
            Neural Networks 43 (2013) 72-83.'
"""

from numpy import inf, array, matrix, diag, multiply, ones, asarray
from numpy.random import randint
from numpy.linalg import solve
from .density_ratio import DensityRatio, KernelInfo
from .helpers import to_numpy_matrix
from .uLSIF import compute_kernel_Gaussian


def RuLSIF(x, y, alpha, sigma_range, lambda_range, kernel_num=100, verbose=True):
    """
    Estimation of the alpha-Relative Density Ratio p(x)/p_alpha(y) by RuLSIF
    (Relative Unconstrained Least-Square Importance Fitting)

    p_alpha(y) = alpha * p(y) + (1 - alpha) * q(y)

    Arguments:
        x (numpy.matrix): sample from p(x).
        y (numpy.matrix): sample from p(x).
        sigma_range (list<float>): search range of Gaussian kernel bandwidth.
        lambda_range (list<float>): search range of regularization parameter.
        kernel_num (int): number of kernels. (default 100)
        verbose (bool): indicator to print messages (default True)

    Returns:
        densratio.DensityRatio object which has `compute_density_ratio()`.
    """

    if alpha < 0:
        raise TypeError('Alpha must be positive.')

    nx = x.shape[0]
    ny = y.shape[0]
    kernel_num = min(kernel_num, nx)
    centers = x[randint(nx, size=kernel_num)]

    if verbose:
        print("RuLSIF starting...")

    if sigma_range.size == 1 and lambda_range.size == 1:
        sigma = sigma_range[0]
        lambda_ = lambda_range[0]
    else:
        if verbose:
            print("Searching for the optimal sigma and lambda...")

        opt_params = search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range, verbose)
        sigma = opt_params["sigma"]
        lambda_ = opt_params["lambda"]

        if verbose:
            print("Found optimal sigma = %.3f, lambda = %.3f." % (sigma, lambda_))

    if verbose:
        print("Optimizing alpha...")

    phi_x = compute_kernel_Gaussian(x, centers, sigma)
    phi_y = compute_kernel_Gaussian(y, centers, sigma)
    H = alpha * (phi_x.T.dot(phi_x) / nx) + (1 - alpha) * (phi_y.T.dot(phi_y) / ny)
    h = phi_x.mean(axis=0).T
    theta = asarray(solve(H + diag(array(lambda_).repeat(kernel_num)), h)).ravel()
    theta[theta < 0] = 0

    def alpha_density_ratio(x):
        x = to_numpy_matrix(x)
        phi_x = compute_kernel_Gaussian(x, centers, sigma)
        alpha_density_ratio = asarray(phi_x.dot(matrix(theta).T)).ravel()
        return alpha_density_ratio

    kernel_info = KernelInfo(kernel_type="Gaussian RBF", kernel_num=kernel_num, sigma=sigma, centers=centers)
    result = DensityRatio(method="RuLSIF", alpha=alpha, lambda_=lambda_, kernel_info=kernel_info, compute_density_ratio=alpha_density_ratio)

    if verbose:
        print("RuLSIF completed.")

    return result


def search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range, verbose):
    nx = x.shape[0]
    ny = y.shape[0]
    n_min = min(nx, ny)
    kernel_num = centers.shape[0]

    score_new = inf
    sigma_new = 0
    lambda_new = 0

    for sigma in sigma_range:
        phi_x = compute_kernel_Gaussian(x, centers, sigma)
        phi_y = compute_kernel_Gaussian(y, centers, sigma)
        H = alpha * (phi_x.T.dot(phi_x) / nx) + (1 - alpha) * (phi_y.T.dot(phi_y) / ny)
        h = phi_x.mean(axis=0).T
        phi_x = phi_x[:n_min].T
        phi_y = phi_y[:n_min].T
        for lambda_ in lambda_range:
            B = H + diag(array(lambda_ * (ny - 1) / ny).repeat(kernel_num))
            B_inv_X = solve(B, phi_y)
            X_B_inv_X = multiply(phi_y, B_inv_X)
            denom = (ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X)).A1
            B0 = solve(B, h.dot(matrix(ones(n_min)))) + B_inv_X.dot(diag(h.T.dot(B_inv_X).A1 / denom))
            B1 = solve(B, phi_x) + B_inv_X.dot(diag(ones(kernel_num).dot(multiply(phi_x, B_inv_X)).A1))
            B2 = (ny - 1) * (nx * B0 - B1) / (ny * (nx - 1))
            B2[B2 < 0] = 0
            r_y = multiply(phi_y, B2).sum(axis=0).T
            r_x = multiply(phi_x, B2).sum(axis=0).T

            score = (r_y.T.dot(r_y).A1 / 2 - r_x.sum(axis=0)) / n_min
            if score < score_new:
                if verbose:
                    print("sigma = %.3f, lambda = %.3f, score = %.3f" % (sigma, lambda_, score))
                score_new = score
                sigma_new = sigma
                lambda_new = lambda_

    return {"sigma": sigma_new, "lambda": lambda_new}