import unittest

from scipy.stats import norm, multivariate_normal
from numpy import linspace, mgrid
from .context import densratio


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_alphadensratio_1d_1(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = norm.rvs(size=200, loc=0, scale=1./2, random_state=71)
        result = densratio(x, y, alpha=0)
        self.assertIsNotNone(result)
        density_ratio = result.compute_density_ratio(linspace(-1, 3))
        self.assertTrue((density_ratio >= 0).all())

    def test_alphadensratio_1d_2(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = norm.rvs(size=200, loc=0, scale=1./2, random_state=71)
        result = densratio(x, y, alpha=0.5)
        self.assertIsNotNone(result)
        density_ratio = result.compute_density_ratio(linspace(-1, 3))
        self.assertTrue((density_ratio >= 0).all())

    def test_alphadensratio_1d_2(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = norm.rvs(size=200, loc=0, scale=1./2, random_state=71)
        result = densratio(x, y, alpha=0.5)
        self.assertIsNotNone(result)
        density_ratio = result.compute_density_ratio(linspace(-1, 3))
        self.assertTrue((density_ratio >= 0).all())

    def test_alphadensratio_1d_3(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = norm.rvs(size=200, loc=0, scale=1./2, random_state=71)
        result = densratio(x, y, alpha=1)
        self.assertIsNotNone(result)
        density_ratio = result.compute_density_ratio(linspace(-1, 3))
        self.assertTrue((density_ratio >= 0).all())

    def test_alphadensratio_2d(self):
        x = multivariate_normal.rvs(size=300, mean=[1, 1], cov=[[1./8, 0], [0, 2]], random_state=71)
        y = multivariate_normal.rvs(size=300, mean=[1, 1], cov=[[1./2, 0], [0, 2]], random_state=71)
        result = densratio(x, y, alpha=0.5)
        self.assertIsNotNone(result)
        space_range = slice(-1, 3, 50j)
        space_2d = mgrid[space_range, space_range].reshape(2, -1).T
        density_ratio = result.compute_density_ratio(space_2d)
        self.assertTrue((density_ratio >= 0).all())

    def test_densratio_dimension_error(self):
        x = norm.rvs(size=200, loc=0, scale=1./8, random_state=71)
        y = multivariate_normal.rvs(size=300, mean=[1, 1], cov=[[1./2, 0], [0, 2]], random_state=71)
        with self.assertRaises(ValueError):
            densratio(x, y, alpha=0.7)


if __name__ == '__main__':
    unittest.main()
