"""
Relative Unconstrained Least-Squares Fitting (RuLSIF): A Python Implementation
References:
    'Change-point detection in time-series data by relative density-ratio estimation'
        Song Liu, Makoto Yamada, Nigel Collier and Masashi Sugiyama,
        Neural Networks 43 (2013) 72-83.

    'A Least-squares Approach to Direct Importance Estimation'
        Takafumi Kanamori, Shohei Hido, and Masashi Sugiyama,
        Journal of Machine Learning Research 10 (2009) 1391-1445.
"""

from numpy import inf, array, exp, matrix, diag, multiply, ones, asarray, log
from numpy.random import randint
from numpy.linalg import norm, solve
from .density_ratio import DensityRatio, KernelInfo
from .helpers import to_numpy_matrix


def RuLSIF(x, y, alpha, sigma_range, lambda_range, kernel_num=100, verbose=True):
    """
    Estimation of the alpha-Relative Density Ratio p(x)/p_alpha(x) by RuLSIF
    (Relative Unconstrained Least-Square Importance Fitting)

    p_alpha(x) = alpha * p(x) + (1 - alpha) * q(x)

    Arguments:
        x (numpy.matrix): Sample from p(x).
        y (numpy.matrix): Sample from q(x).
        alpha (float): Mixture parameter.
        sigma_range (list<float>): Search range of Gaussian kernel bandwidth.
        lambda_range (list<float>): Search range of regularization parameter.
        kernel_num (int): Number of kernels. (Default 100)
        verbose (bool): Indicator to print messages (Default True)

    Returns:
        densratio.DensityRatio object which has `compute_density_ratio()`.
    """

    # Number of samples.
    nx = x.shape[0]
    ny = y.shape[0]

    # Number of kernel functions.
    kernel_num = min(kernel_num, nx)

    # Randomly take a subset of x, to identify centers for the kernels.
    centers = x[randint(nx, size=kernel_num)]

    if verbose:
        print("RuLSIF starting...")

    if len(sigma_range) == 1 and len(lambda_range) == 1:
        sigma = sigma_range[0]
        lambda_ = lambda_range[0]
    else:
        if verbose:
            print("Searching for the optimal sigma and lambda...")

        # Grid-search cross-validation for optimal kernel and regularization parameters.
        opt_params = search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range, verbose)
        sigma = opt_params["sigma"]
        lambda_ = opt_params["lambda"]

        if verbose:
            print("Found optimal sigma = {:.3f}, lambda = {:.3f}.".format(sigma, lambda_))

    if verbose:
        print("Optimizing theta...")

    phi_x = compute_kernel_Gaussian(x, centers, sigma)
    phi_y = compute_kernel_Gaussian(y, centers, sigma)
    H = alpha * (phi_x.T.dot(phi_x) / nx) + (1 - alpha) * (phi_y.T.dot(phi_y) / ny)
    h = phi_x.mean(axis=0).T
    theta = asarray(solve(H + diag(array(lambda_).repeat(kernel_num)), h)).ravel()

    # No negative coefficients.
    theta[theta < 0] = 0

    # Compute the alpha-relative density ratio, at the given coordinates.
    def alpha_density_ratio(coordinates):
        # Evaluate the kernel at these coordinates, and take the dot-product with the weights.
        coordinates = to_numpy_matrix(coordinates)
        phi_x = compute_kernel_Gaussian(coordinates, centers, sigma)
        alpha_density_ratio = asarray(phi_x.dot(matrix(theta).T)).ravel()

        return alpha_density_ratio

    # Compute the approximate alpha-relative PE-divergence, given samples x and y from the respective distributions.
    def alpha_PE_divergence(x, y):
        # This is Y, in Reference 1.
        x = to_numpy_matrix(x)

        # Obtain alpha-relative density ratio at these points.
        g_x = alpha_density_ratio(x)

        # This is Y', in Reference 1.
        y = to_numpy_matrix(y)

        # Obtain alpha-relative density ratio at these points.
        g_y = alpha_density_ratio(y)

        # Compute the alpha-relative PE-divergence as given in Reference 1.
        n = x.shape[0]
        divergence = (-alpha*(g_x.T.dot(g_x))/2 - (1-alpha)*(g_y.T.dot(g_y))/2 + g_x.sum(axis=0))/n - 1./2
        return divergence

    # Compute the approximate alpha-relative KL-divergence, given samples x and y from the respective distributions.
    def alpha_KL_divergence(x, y):
        # This is Y, in Reference 1.
        x = to_numpy_matrix(x)

        # Obtain alpha-relative density ratio at these points.
        g_x = alpha_density_ratio(x)

        # Compute the alpha-relative KL-divergence.
        n = x.shape[0]
        divergence = log(g_x).sum(axis=0) / n
        return divergence

    alpha_PE = alpha_PE_divergence(x, y)
    alpha_KL = alpha_KL_divergence(x, y)

    if verbose:
        print("Approximate alpha-relative PE-divergence = {:03.2f}".format(alpha_PE))
        print("Approximate alpha-relative KL-divergence = {:03.2f}".format(alpha_KL))

    kernel_info = KernelInfo(kernel_type="Gaussian", kernel_num=kernel_num, sigma=sigma, centers=centers)
    result = DensityRatio(method="RuLSIF", alpha=alpha, theta=theta, lambda_=lambda_, alpha_PE=alpha_PE, alpha_KL=alpha_KL,
                          kernel_info=kernel_info, compute_density_ratio=alpha_density_ratio)

    if verbose:
        print("RuLSIF completed.")

    return result


# Grid-search cross-validation for the optimal parameters sigma and lambda by leave-one-out cross-validation. See Reference 2.
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

            # Squared loss of RuLSIF, without regularization term.
            # Directly related to the negative of the PE-divergence.
            score = (r_y.T.dot(r_y).A1 / 2 - r_x.sum(axis=0)) / n_min

            if verbose:
                print("sigma = %.5f, lambda = %.5f, score = %.5f" % (sigma, lambda_, score))

            if score < score_new:
                score_new = score
                sigma_new = sigma
                lambda_new = lambda_

    return {"sigma": sigma_new, "lambda": lambda_new}


# Returns a 2D numpy matrix of kernel evaluated at the gridpoints with coordinates from x_list and y_list.
def compute_kernel_Gaussian(x_list, y_list, sigma):
    result = [[kernel_Gaussian(x, y, sigma) for y in y_list] for x in x_list]
    result = matrix(result)
    return result


# Returns the Gaussian kernel evaluated at (x, y) with parameter sigma.
def kernel_Gaussian(x, y, sigma):
    return exp(- (norm(x - y) ** 2) / (2 * sigma * sigma))
