"""
Kullback-Leibler Importance Estimation Procedure (KLIEP).
"""

from numbers import Number

from numpy import arange, asarray, inf, log, mean, ones
from numpy.random import choice, permutation

from .RuLSIF import compute_kernel_Gaussian
from .density_ratio import DensityRatio, KernelInfo


def KLIEP(x, y, sigma="auto", kernel_num=100, fold=5, verbose=True):
    nx = x.shape[0]
    kernel_num = min(kernel_num, nx)
    centers = x[choice(nx, size=kernel_num, replace=False)]

    if verbose:
        print("KLIEP starting...")

    sigma = _select_sigma(x, y, centers, sigma, fold, verbose)

    if verbose:
        print("Optimizing kernel weights...")

    phi_x = compute_kernel_Gaussian(x, centers, sigma)
    phi_y = compute_kernel_Gaussian(y, centers, sigma)
    kernel_weights = KLIEP_optimize_alpha(phi_x, phi_y)

    kernel_info = KernelInfo(kernel_type="Gaussian", kernel_num=kernel_num, sigma=sigma, centers=centers)
    result = DensityRatio(method="KLIEP", alpha=None, theta=kernel_weights, lambda_=None, alpha_PE=None,
                          alpha_KL=None, kernel_info=kernel_info, fold=fold)

    if verbose:
        print("KLIEP completed.")

    return result


def _select_sigma(x, y, centers, sigma, fold, verbose):
    if isinstance(sigma, str):
        if sigma != "auto":
            raise TypeError("Invalid value for sigma.")

        if verbose:
            print("Searching for the optimal sigma...")

        sigma = KLIEP_search_sigma(x, y, centers, fold, verbose)

        if verbose:
            print("Found optimal sigma = {:.5f}.".format(sigma))

        return sigma

    if isinstance(sigma, Number):
        return float(sigma)

    sigma_range = asarray(sigma).ravel()
    if sigma_range.size == 1:
        return float(sigma_range[0])

    if verbose:
        print("Searching for the optimal sigma...")

    sigma = KLIEP_search_sigma_list(x, y, centers, sigma_range, fold, verbose)

    if verbose:
        print("Found optimal sigma = {:.5f}.".format(sigma))

    return sigma


def KLIEP_search_sigma(x, y, centers, fold, verbose=True):
    sigma = 10.0
    score = -inf

    for digit_position in range(0, -6, -1):
        for _ in range(9):
            sigma_new = sigma - 10.0 ** digit_position
            score_new = KLIEP_compute_score_cv(sigma_new, x, y, centers, fold)
            if score_new <= score:
                break

            score = score_new
            sigma = sigma_new

            if verbose:
                print("  sigma = {:.5f}, score = {:.6f}".format(sigma, score))

    return sigma


def KLIEP_search_sigma_list(x, y, centers, sigma_list, fold, verbose=True):
    sigma = None
    score = -inf

    for sigma_new in sigma_list:
        score_new = KLIEP_compute_score_cv(sigma_new, x, y, centers, fold)
        if score_new > score:
            score = score_new
            sigma = float(sigma_new)

            if verbose:
                print("  sigma = {:.5f}, score = {:.6f}".format(sigma, score))

    return sigma


def KLIEP_compute_score_cv(sigma, x, y, centers, fold):
    phi_x = compute_kernel_Gaussian(x, centers, sigma)
    phi_y = compute_kernel_Gaussian(y, centers, sigma)
    mean_phi_y = phi_y.mean(axis=0)

    nx = x.shape[0]
    cv_split = permutation(nx) % fold

    scores = []
    for fold_index in range(fold):
        alpha = KLIEP_optimize_alpha(phi_x[cv_split != fold_index, :], mean_phi_y=mean_phi_y)
        scores.append(KLIEP_compute_score(phi_x[cv_split == fold_index, :], alpha))

    return mean(scores)


def KLIEP_optimize_alpha(phi_x, phi_y=None, mean_phi_y=None):
    a = asarray(phi_x)
    if phi_y is None:
        if mean_phi_y is None:
            raise ValueError("mean_phi_y must be provided when phi_y is omitted.")
        b = asarray(mean_phi_y).ravel()
    else:
        b = asarray(phi_y).mean(axis=0).ravel()

    c = b / (b @ b)
    alpha = compute_next_alpha(ones(a.shape[1]), b, c)
    score = KLIEP_compute_score(a, alpha)

    for epsilon in 10.0 ** arange(3, -4, -1):
        for _ in range(100):
            alpha_new = alpha + epsilon * a.T @ (1 / (a @ alpha))
            alpha_new = compute_next_alpha(alpha_new, b, c)
            score_new = KLIEP_compute_score(a, alpha_new)

            if score_new <= score:
                break

            alpha = alpha_new
            score = score_new

    return alpha


def compute_next_alpha(alpha, b, c=None):
    if c is None:
        c = b / (b @ b)

    alpha = alpha + (1 - b @ alpha) * c
    alpha[alpha < 0] = 0
    return alpha / (b @ alpha)


def KLIEP_compute_score(phi, alpha):
    return mean(log(phi @ alpha))
