A Python Package for Density Ratio Estimation
================
Koji Makiyama (@hoxo-m), Ameya Daigavane (@ameya98), and Krzysztof
Mierzejewski (@mierzejk)

<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- badges: start -->

[![CI](https://github.com/hoxo-m/densratio_py/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hoxo-m/densratio_py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/densratio.svg)](https://pypi.org/project/densratio/)
[![PyPI](https://img.shields.io/pypi/dm/densratio.svg)](https://pypi.org/project/densratio/)
<!-- badges: end -->

## 1. Overview

**Density ratio estimation** is described as follows: for given two data
samples `x1` and `x2` from unknown distributions `p(x)` and `q(x)`
respectively, estimate `w(x) = p(x) / q(x)`, where `x1` and `x2` are
d-dimensional real numbers.

The estimated density ratio function `w(x)` can be used in many
applications such as the inlier-based outlier detection \[1\] and
covariate shift adaptation \[2\]. Other useful applications for density
ratio estimation were summarized by Sugiyama et al. (2012) in \[3\].

The package **densratio** provides `densratio()` and method-specific
wrappers `uLSIF()`, `RuLSIF()`, and `KLIEP()`. Each estimator returns an
object with `compute_density_ratio()` for evaluating the learned density
ratio on new samples. The default method is `uLSIF`, matching the R
package API.

Further, the alpha-relative density ratio
`p(x)/(alpha * p(x) + (1 - alpha) * q(x))` (where alpha is in the range
\[0, 1\]) can also be estimated. When alpha is 0, this reduces to the
ordinary density ratio `w(x)`. The alpha-relative PE-divergence and
KL-divergence between `p(x)` and `q(x)` are also computed.

![](README_files/figure-gfm/compare-true-estimate-1.png)<!-- -->

For example,

``` python
import numpy as np
from scipy.stats import norm
from densratio import densratio

np.random.seed(1)
x = norm.rvs(size=500, loc=0, scale=1./8)
y = norm.rvs(size=500, loc=0, scale=1./2)
alpha = 0.1
densratio_obj = densratio(x, y, method="RuLSIF", alpha=alpha)
print(densratio_obj)
```

gives the following output:

    #> Method: RuLSIF
    #> 
    #> Alpha: 0.1
    #> 
    #> Kernel Information:
    #>   Kernel type: Gaussian
    #>   Number of kernels: 100
    #>   Bandwidth(sigma): 0.1
    #>   Centers: array([[-0.09591373],..
    #> 
    #> Kernel Weights (theta):
    #>   array([0.04990797, 0.0550548 , 0.04784736, 0.04951904, 0.04840418,..
    #> 
    #> Regularization Parameter (lambda): 0.1
    #> 
    #> Alpha-Relative PE-Divergence: 0.618794133598705
    #> 
    #> Alpha-Relative KL-Divergence: 0.7037648129307483
    #> 
    #> Function to Estimate Density Ratio:
    #>   compute_density_ratio(x)
    #> 

In this case, the true density ratio `w(x)` is known, so we can compare
`w(x)` with the estimated density ratio `w-hat(x)`. The code below gives
the plot shown above.

``` python
from matplotlib import pyplot as plt
from numpy import linspace

def true_alpha_density_ratio(sample):
    return norm.pdf(sample, 0, 1./8) / (alpha * norm.pdf(sample, 0, 1./8) + (1 - alpha) * norm.pdf(sample, 0, 1./2))

def estimated_alpha_density_ratio(sample):
    return densratio_obj.compute_density_ratio(sample)

sample_points = np.linspace(-1, 3, 400)
plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
plt.legend()
plt.show()
```

## 2. Installation

You can install the package from
[PyPI](https://pypi.org/project/densratio/).

``` :sh
$ pip install densratio
```

**densratio** supports Python 3.10 or later.

Also, you can install the package from
[GitHub](https://github.com/hoxo-m/densratio_py).

``` :sh
$ pip install git+https://github.com/hoxo-m/densratio_py.git
```

The source code for **densratio** package is available on GitHub at
<https://github.com/hoxo-m/densratio_py>.

## 3. Details

### 3.1. Basics

The package provides `densratio()`. The function returns an object that
has a function to compute estimated density ratio.

For data samples `x` and `y`,

``` python
from scipy.stats import norm
from densratio import densratio

x = norm.rvs(size = 200, loc = 1, scale = 1./8)
y = norm.rvs(size = 200, loc = 1, scale = 1./2)
result = densratio(x, y)
```

In this case, `result.compute_density_ratio()` can compute estimated
density ratio.

``` python
from matplotlib import pyplot as plt

density_ratio = result.compute_density_ratio(y)

plt.plot(y, density_ratio, "o")
plt.xlabel("x")
plt.ylabel("Density Ratio")
plt.show()
```

![](README_files/figure-gfm/plot-estimated-density-ratio-3.png)<!-- -->

### 3.2. Methods

The package estimates density ratios with Gaussian-kernel direct density
ratio estimators. Use the `method` argument of `densratio()` or call the
method-specific wrappers directly:

-   `uLSIF(x, y, ...)` estimates the ordinary density ratio
    `p(x) / q(x)` by unconstrained Least-Squares Importance Fitting.
    This is the default for `densratio(x, y)`.
-   `RuLSIF(x, y, alpha=0.1, ...)` estimates the alpha-relative density
    ratio `p(x) / (alpha * p(x) + (1 - alpha) * q(x))`. It also reports
    alpha-relative PE-divergence and KL-divergence.
-   `KLIEP(x, y, fold=5, ...)` estimates the ordinary density ratio by
    Kullback-Leibler Importance Estimation Procedure. It uses
    cross-validation over `sigma` when a search range is provided.

For example:

``` python
ordinary = densratio(x, y)
relative = densratio(x, y, method="RuLSIF", alpha=0.1)
kliep = densratio(x, y, method="KLIEP", sigma=[0.1, 0.3, 1.0], fold=5)
```

All methods represent the density ratio with a linear Gaussian RBF
kernel model:

`w(x) = theta1 * K(x, c1) + theta2 * K(x, c2) + ... + thetab * K(x, cb)`
where `K(x, c) = exp(- ||x - c||^2 / (2 * sigma ^ 2))` is the Gaussian
RBF kernel.

`densratio()` performs the following:

-   Decides kernel parameter `sigma` by cross-validation.
-   Optimizes for kernel weights `theta`.
-   For RuLSIF, computes the alpha-relative PE-divergence and
    KL-divergence from the learned alpha-relative ratio.

Kernel centers are selected at random from `x`, the numerator sample.
Set `numpy.random.seed(...)` before fitting when reproducible centers are
needed.

As the result, you can obtain `compute_density_ratio()`, which will
compute the estimated density ratio at the passed coordinates.

### 3.3. Result and Parameter Settings

`densratio()` outputs the result like as follows:

    #> Method: uLSIF
    #> 
    #> Alpha: 0
    #> 
    #> Kernel Information:
    #>   Kernel type: Gaussian
    #>   Number of kernels: 100
    #>   Bandwidth(sigma): 0.1
    #>   Centers: array([[0.92113356],..
    #> 
    #> Kernel Weights (theta):
    #>   array([0.08848922, 0.03377533, 0.0753727 , 0.06141277, 0.02543963,..
    #> 
    #> Regularization Parameter (lambda): 1.0
    #> 
    #> Alpha-Relative PE-Divergence: 0.9635169300831041
    #> 
    #> Alpha-Relative KL-Divergence: 0.838826626547327
    #> 
    #> Function to Estimate Density Ratio:
    #>   compute_density_ratio(x)
    #> 

-   **Method** is `uLSIF`, `RuLSIF`, or `KLIEP`.
-   **Kernel type** is fixed as Gaussian RBF.
-   **Number of kernels** is the number of kernels in the linear model.
    You can change by setting `kernel_num` parameter. In default,
    `kernel_num = 100`.
-   **Bandwidth(sigma)** is the Gaussian kernel bandwidth. In default,
    `sigma = "auto"`, the algorithm automatically select an optimal
    value by cross validation. If you set `sigma` a number, that will be
    used. If you set `sigma` a numeric array, the algorithm select an
    optimal value in them by cross validation.
-   **Centers** are centers of Gaussian kernels in the linear model.
    These are selected at random from the data sample `x` underlying a
    numerator distribution `p(x)`. You can find the whole values in
    `result.kernel_info.centers`.
-   **Kernel weights(theta)** are theta parameters in the linear kernel
    model. You can find these values in `result.theta`, or
    `result.kernel_weights` for R-style naming.
-   **Regularization parameter(lambda)** is used by `uLSIF` and
    `RuLSIF`. It is not used by `KLIEP`.
-   **Fold** is used by `KLIEP` cross-validation.
-   **The function to estimate the density ratio** is named
    `compute_density_ratio()`.

### 3.4. Setting Gaussian kernel calculation engine

When working out Gaussian kernels, linear algebra calculations can be done either with `numpy` or `numba` packages. The `densratio.set_compute_kernel_target` function accepts a single `str` argument to globally select a specified engine:
- `numpy` - [**numpy** broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting) optimized. It must be noted the underlying BLAS library (e.g. Intel's MKL) can take advantage of [multi threading model](https://software.intel.com/content/www/us/en/develop/documentation/mkl-linux-developer-guide/top/managing-performance-and-memory/improving-performance-with-threading/using-additional-threading-control.html).
- `cpu` - [**numba** generalized universal function single thread](https://numba.pydata.org/numba-doc/latest/user/vectorize.html#the-guvectorize-decorator) optimized.
- `parallel` - [**numba** generalized universal function multi thread](https://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html#numba.guvectorize) optimized. Please be advised all [threading layer specifics](https://numba.pydata.org/numba-doc/latest/user/threading-layer.html) apply.

`densratio` defaults to `cpu` when `numba` is available, or `numpy` otherwise.

Although `numba` is not a requirement of `densratio_py`, version `0.45.1` or later is necessary to set the calculation engine to `cpu` or `parallel`.

## 4. Multi Dimensional Data Samples

So far, we have deal with one-dimensional data samples `x` and `y`.
`densratio()` allows to input multidimensional data samples as
`numpy.ndarray` or `numpy.matrix`, as long as their dimensions are the
same.

For example,

``` python
from scipy.stats import multivariate_normal
from densratio import densratio

np.random.seed(1)
x = multivariate_normal.rvs(size=3000, mean=[1, 1], cov=[[1. / 8, 0], [0, 1. / 8]])
y = multivariate_normal.rvs(size=3000, mean=[1, 1], cov=[[1. / 2, 0], [0, 1. / 2]])
alpha = 0
densratio_obj = densratio(x, y, method="RuLSIF", alpha=alpha, sigma_range=[0.1, 0.3, 0.5, 0.7, 1], lambda_range=[0.01, 0.02, 0.03, 0.04, 0.05])
print(densratio_obj)
```

gives the following output:

    #> Method: RuLSIF
    #> 
    #> Alpha: 0
    #> 
    #> Kernel Information:
    #>   Kernel type: Gaussian
    #>   Number of kernels: 100
    #>   Bandwidth(sigma): 0.3
    #>   Centers: array([[1.01477443, 1.38864061],..
    #> 
    #> Kernel Weights (theta):
    #>   array([0.06151164, 0.08012094, 0.10467369, 0.13868176, 0.14917063,..
    #> 
    #> Regularization Parameter (lambda): 0.04
    #> 
    #> Alpha-Relative PE-Divergence: 0.653615870855595
    #> 
    #> Alpha-Relative KL-Divergence: 0.6214285743087565
    #> 
    #> Function to Estimate Density Ratio:
    #>   compute_density_ratio(x)
    #> 

In this case, as well, we can compare the true density ratio with the
estimated density ratio.

``` python
from matplotlib import pyplot as plt
from numpy import linspace, dstack, meshgrid, concatenate

def true_alpha_density_ratio(x):
    return multivariate_normal.pdf(x, [1., 1.], [[1. / 8, 0], [0, 1. / 8]]) / \
           (alpha * multivariate_normal.pdf(x, [1., 1.], [[1. / 8, 0], [0, 1. / 8]]) + (1 - alpha) * multivariate_normal.pdf(x, [1., 1.], [[1. / 2, 0], [0, 1. / 2]]))

def estimated_alpha_density_ratio(x):
    return densratio_obj.compute_density_ratio(x)

range_ = np.linspace(0, 2, 200)
grid = np.concatenate(np.dstack(np.meshgrid(range_, range_)))
levels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.contourf(range_, range_, true_alpha_density_ratio(grid).reshape(200, 200), levels)
#> <matplotlib.contour.QuadContourSet object at 0x0000022E950202E0>
plt.colorbar()
#> <matplotlib.colorbar.Colorbar object at 0x0000022E9500DA80>
plt.title("True Alpha-Relative Density Ratio")
plt.subplot(1, 2, 2)
plt.contourf(range_, range_, estimated_alpha_density_ratio(grid).reshape(200, 200), levels)
#> <matplotlib.contour.QuadContourSet object at 0x0000022E942C8EE0>
plt.colorbar()
#> <matplotlib.colorbar.Colorbar object at 0x0000022E95095150>
plt.title("Estimated Alpha-Relative Density Ratio")
plt.show()
```

![](README_files/figure-gfm/compare-2d-5.png)<!-- -->

## 5. References

\[1\] Hido, S., Tsuboi, Y., Kashima, H., Sugiyama, M., & Kanamori, T.
**Statistical outlier detection using direct density ratio estimation.**
Knowledge and Information Systems 2011.

\[2\] Sugiyama, M., Nakajima, S., Kashima, H., von Bunau, P. & Kawanabe,
M. **Direct importance estimation with model selection and its
application to covariate shift adaptation.** NIPS 2007.

\[3\] Sugiyama, M., Suzuki, T. & Kanamori, T. **Density Ratio Estimation
in Machine Learning.** Cambridge University Press 2012.

\[4\] Liu, S., Yamada, M., Collier, N., & Sugiyama, M. **Change-Point
Detection in Time-Series Data by Relative Density-Ratio Estimation**
Neural Networks, 2013.

## 6. Related Work

-   densratio for R <https://github.com/hoxo-m/densratio>
-   pykliep <https://github.com/srome/pykliep>
