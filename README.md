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

x = norm.rvs(size = 200, loc = 0, scale = 1./8, random_state = 71)
y = norm.rvs(size = 200, loc = 0, scale = 1./2, random_state = 71)
result = densratio(x, y)
print(result)
```


```
#> ################## Start uLSIF ##################
#> Searching optimal sigma and lambda...
#>   sigma = 0.316, lambda = 0.001, score = -0.541
#>   sigma = 0.316, lambda = 0.003, score = -1.192
#> Found optimal sigma = 0.316, lambda = 0.003.
#> Optimizing alpha...
#> End.
#> ################## Finished uLSIF ###############
#> Method: uLSIF
#> 
#> Kernel Information:
#>   Kernel type: Gaussian RBF
#>   Number of kernels: 100
#>   Bandwidth(sigma): 0.316227766017
#>   Centers: array([-0.09598839, -0.11855759, -0.05781936, -0.18525961, 0.04098506,..
#> 
#> Kernel Weights(alpha):
#>   array([ 0.2880579 , 0.49622356, 0.48327207, 0. , 0.13081916,..
#> 
#> Regularization Parameter(lambda): 0.00316227766017
#> 
#> The Function to Estimate Density Ratio:
#>   compute_density_ratio(x)
```
