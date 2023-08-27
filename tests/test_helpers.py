import unittest

import numpy as np
from numpy import array
from numpy.testing import assert_array_equal
from pandas import DataFrame
from .context import helpers


# noinspection PyMethodMayBeStatic
class BasicTestSuite(unittest.TestCase):
    """Tests for Helper Functions."""

    def test_is_numeric(self):
        x = 1
        y = 1.0
        z = "1"
        self.assertTrue(helpers.is_numeric(x))
        self.assertTrue(helpers.is_numeric(y))
        self.assertFalse(helpers.is_numeric(z))

    def test_to_ndarray_list(self):
        x = [1,2,3]
        x = helpers.to_ndarray(x)
        assert_array_equal(x, array([1,2,3]).reshape(-1, 1))

    def test_to_ndarray_list2(self):
        x = [[1,2],[3,4]]
        x = helpers.to_ndarray(x)
        assert_array_equal(x, array([[1,2],[3,4]]))

    def test_to_ndarray_ndarray(self):
        x = array([1,2,3])
        x = helpers.to_ndarray(x)
        assert_array_equal(x, array([1,2,3]).reshape(-1, 1))

    def test_to_ndarray_ndarray2(self):
        x = array([[1,2],[3,4]])
        x = helpers.to_ndarray(x)
        assert_array_equal(x, array([[1,2],[3,4]]))

    # def test_to_ndarray_matrix(self):
    #     x = matrix([[1,2],[3,4]])
    #     x = helpers.to_ndarray(x)
    #     assert_array_equal(x, array([[1,2],[3,4]]))

    def test_to_ndarray_pandas_DataFrame(self):
        x = DataFrame([[1,2],[3,4]])
        x = helpers.to_ndarray(x)
        assert_array_equal(x, array([[1,2],[3,4]]))

    def test_alpha_normalize_0d_input(self):
        with self.assertRaises(ValueError):
            helpers.alpha_normalize(np.array(None), 1E-3)

    def test_alpha_normalize_2d_input(self):
        with self.assertRaises(ValueError):
            helpers.alpha_normalize(np.arange(4).reshape(2, -1), 1E-3)

    def test_alpha_normalize_zero_alpha(self):
        values = np.random.rand(5)
        assert_array_equal(values, helpers.alpha_normalize(values, 0))

    def test_alpha_normalize_negative_alpha(self):
        values = np.random.rand(5)
        with self.assertWarns(RuntimeWarning):
            assert_array_equal(values, helpers.alpha_normalize(values, -1E-3))

    def test_alpha_normalize_lower_changed(self):
        alpha = .05
        lower, upper = np.arange(0, 1, alpha), np.arange(1, 21)
        normalized_values = helpers.alpha_normalize(np.concatenate((lower, upper)), alpha)
        self.assertRaises(AssertionError, assert_array_equal, lower, normalized_values[:lower.size])
        assert_array_equal(upper, normalized_values[-upper.size:])

    def test_alpha_normalize_equal_unique_count(self):
        alpha = .05
        values = np.concatenate((np.arange(0, 1, alpha), np.arange(1, 21)))
        normalized_values = helpers.alpha_normalize(values, alpha)
        self.assertEqual(np.unique(values).size, np.unique(normalized_values).size)

    def test_alpha_normalize_same_argsort(self):
        alpha = .05
        values = np.concatenate((np.arange(0, 1, alpha), np.arange(1, 21)))
        np.random.shuffle(values)
        normalized_values = helpers.alpha_normalize(values, alpha)
        self.assertTrue((np.argsort(values) == np.argsort(normalized_values)).all())

    def test_alpha_normalize_alpha_warning(self):
        values = np.array([0, np.nextafter(0, 1), 1])
        self.assertWarns(RuntimeWarning, helpers.alpha_normalize, values, .5)

    def test_alpha_normalize_nonpositive_warning(self):
        values = np.array([0, np.nextafter(0, 1), 1])
        self.assertWarns(RuntimeWarning, helpers.alpha_normalize, values, values[1])


if __name__ == '__main__':
    unittest.main()
