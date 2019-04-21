# -*- coding: utf-8 -*-

import unittest

from scipy.stats import norm, multivariate_normal
from numpy import linspace
from .context import densratio

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_densratio_1d(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = norm.rvs(size=200, loc=0, scale=1./2, random_state=71)
        result = densratio(x, y, alpha=0)
        self.assertIsNotNone(result)
        print(result)
        density_ratio = result.compute_density_ratio(linspace(-1, 3))
        # print(density_ratio)

    def test_densratio_2d(self):
        x = multivariate_normal.rvs(size=300, mean=[1, 1], cov=[[1./8, 0], [0, 2]], random_state=71)
        y = multivariate_normal.rvs(size=300, mean=[1, 1], cov=[[1./2, 0], [0, 2]], random_state=71)
        result = densratio(x, y)
        self.assertIsNotNone(result)

    def test_densratio_dimension_error(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = multivariate_normal.rvs(size=300, mean=[1, 1], cov=[[1./2, 0], [0, 2]], random_state=71)
        with self.assertRaises(ValueError):
            densratio(x, y)


if __name__ == '__main__':
    unittest.main()
