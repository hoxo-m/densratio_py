# -*- coding: utf-8 -*-

import unittest

from numpy import array, matrix
from numpy.testing import assert_array_equal
from .context import helpers

class BasicTestSuite(unittest.TestCase):
    """Tests for Helper Functions."""

    def test_is_numeric(self):
        x = 1
        y = 1.0
        z = "1"
        self.assertTrue(helpers.is_numeric(x))
        self.assertTrue(helpers.is_numeric(y))
        self.assertFalse(helpers.is_numeric(z))

    def test_to_numpy_matrix_list(self):
        x = [1,2,3]
        x = helpers.to_numpy_matrix(x)
        assert_array_equal(x, matrix([1,2,3]).T)

    def test_to_numpy_matrix_list2(self):
        x = [[1,2],[3,4]]
        x = helpers.to_numpy_matrix(x)
        assert_array_equal(x, matrix([[1,2],[3,4]]))

    def test_to_numpy_matrix_ndarray(self):
        x = array([1,2,3])
        x = helpers.to_numpy_matrix(x)
        assert_array_equal(x, matrix([1,2,3]).T)

    def test_to_numpy_matrix_ndarray2(self):
        x = array([[1,2],[3,4]])
        x = helpers.to_numpy_matrix(x)
        assert_array_equal(x, matrix([[1,2],[3,4]]))

    def test_to_numpy_matrix_matrix(self):
        x = matrix([[1,2],[3,4]])
        x = helpers.to_numpy_matrix(x)
        assert_array_equal(x, matrix([[1,2],[3,4]]))


if __name__ == '__main__':
    unittest.main()
