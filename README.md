# A Python Package for Density Ratio Estimation
Koji MAKIYAMA (@hoxo_m)  



## 1. Overview

**Density ratio estimation** is described as follows: for given two data samples `x` and `y` from unknown distributions `p(x)` and `q(y)` respectively, estimate `w(x) = p(x) / q(x)`, where `x` and `y` are d-dimensional real numbers.

The estimated density ratio function `w(x)` can be used in many applications such as the inlier-based outlier detection [1] and covariate shift adaptation [2].
Other useful applications about density ratio estimation were summarized by Sugiyama et al. (2012) [3].

The package **densratio** provides a function `densratio()` that returns a result has the function to estimate density ratio `compute_density_ratio()`.

For example, 
a

```python
from scipy.stats import norm
from densratio import densratio

x = norm.rvs(size = 200, loc = 1, scale = 1./8, random_state = 71)
y = norm.rvs(size = 200, loc = 1, scale = 1./2, random_state = 71)
result = densratio(x, y)
print(result)
```


```
#> Method: uLSIF
#> 
#> Kernel Information:
#>   Kernel type: Gaussian RBF
#>   Number of kernels: 100
#>   Bandwidth(sigma): 0.0316227766017
#>   Centers: array([ 0.07676093, 0.15944279, -0.06443216, 0.07535333, 0.12622057,..
#> 
#> Kernel Weights(alpha):
#>   array([ 7.36162341e+00, 4.85725106e+00, 0.00000000e+00,..
#> 
#> Regularization Parameter(lambda): 0.001
#> 
#> The Function to Estimate Density Ratio:
#>   compute_density_ratio(x)
```


## 2. How to Install

You can install the package from GitHub.

```:sh
$ pip install git+https://github.com/hoxo-m/densratio_py.git
```

