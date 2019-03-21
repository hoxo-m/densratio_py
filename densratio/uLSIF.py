# -*- coding: utf-8 -*-

from numpy import inf, exp, array, matrix, diag, multiply, ones
from numpy.random import randint
from numpy.linalg import norm, solve
from .density_ratio import DensityRatio, KernelInfo
from .helpers import to_numpy_matrix

def uLSIF(x, y, sigma_range, lambda_range, kernel_num = 100, verbose = True):
    """Estimate Density Ratio p(x)/q(y) by uLSIF
                                (unconstrained Least-Square Importance Fitting)

    Args:
        x (numpy.matrix): sample from p(x).
        y (numpy.matrix): sample from p(x).
        sigma_range (list<float>): search range of Gaussian kernel bandwidth.
        lambda_range (list<float>): search range of regularization parameter.

    Kwargs:
        kernel_num (int): number of kernels. (default 100)
        verbose (bool): indicator to print messages (deafult True)

    Returns:
        densratio.DensityRatio object which has `compute_density_ratio()`.
    """

    nx = x.shape[0]
    ny = y.shape[0]
    kernel_num = min(kernel_num, nx)
    centers = x[randint(nx, size=kernel_num)]

    if verbose:
        print("uLSIF starting...")

    if sigma_range.size == 1 and lambda_range.size == 1:
        sigma = sigma_range[0]
        lambda_ = lambda_range[0]
    else:
        if verbose:
            print("Searching optimal sigma and lambda...")
        opt_params = search_sigma_and_lambda(x, y, centers, sigma_range, lambda_range, verbose)
        sigma = opt_params["sigma"]
        lambda_ = opt_params["lambda"]
        if verbose:
            print("Found optimal sigma = %.3f, lambda = %.3f." % (sigma, lambda_))

    if verbose:
        print("Optimizing alpha...")
    phi_x = compute_kernel_Gaussian(x, centers, sigma)
    phi_y = compute_kernel_Gaussian(y, centers, sigma)
    H = phi_y.T.dot(phi_y) / ny
    h = phi_x.mean(axis = 0).T
    alpha = solve(H + diag(array(lambda_).repeat(kernel_num)), h).A1
    alpha[alpha < 0] = 0
    if verbose:
        print("End.")

    kernel_info = KernelInfo(kernel_type = "Gaussian RBF",
                    kernel_num = kernel_num, sigma = sigma,
                    centers = centers)

    def compute_density_ratio(x):
        x = to_numpy_matrix(x)
        phi_x = compute_kernel_Gaussian(x, centers, sigma)
        density_ratio = phi_x.dot(matrix(alpha).T).A1
        return density_ratio

    result = DensityRatio(method = "uLSIF", alpha = alpha,
                lambda_ = lambda_, kernel_info = kernel_info,
                compute_density_ratio = compute_density_ratio)

    if verbose:
        print("uLSIF completed.")

    return result


# Grid-searches for the optimal parameters sigma and lambda by cross-validation.
def search_sigma_and_lambda(x, y, centers, sigma_range, lambda_range, verbose):
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
        H = phi_y.T.dot(phi_y) / ny
        h = phi_x.mean(axis = 0).T
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
            r_y = multiply(phi_y, B2).sum(axis = 0).T
            r_x = multiply(phi_x, B2).sum(axis = 0).T

            score = (r_y.T.dot(r_y).A1 / 2 - r_x.sum(axis=0)) / n_min
            if score < score_new:
                if verbose:
                    print("  sigma = %.3f, lambda = %.3f, score = %.3f" % (sigma, lambda_, score))
                score_new = score
                sigma_new = sigma
                lambda_new = lambda_
    return {"sigma": sigma_new, "lambda": lambda_new}



