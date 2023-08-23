import unittest

from numpy import array
from numpy.testing import assert_array_equal
from pandas import DataFrame
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

if __name__ == '__main__':
    unittest.main()
