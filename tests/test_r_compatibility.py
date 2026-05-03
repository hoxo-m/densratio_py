import importlib
import pickle
import unittest

import numpy as np

from densratio import densratio


class RCompatibilityTestSuite(unittest.TestCase):
    """Baseline checks ported from the R package tests."""

    def setUp(self):
        rng = np.random.RandomState(314)
        self.x = rng.normal(loc=1, scale=1.0 / 8.0, size=200)
        self.y = rng.normal(loc=1, scale=1.0 / 2.0, size=200)

    def fit_rulsif(self, *, alpha=0.1, lambda_=0.1):
        np.random.seed(314)
        return densratio(
            self.x,
            self.y,
            sigma_range=[0.1],
            lambda_range=[lambda_],
            alpha=alpha,
            kernel_num=20,
            verbose=False,
        )

    def assert_kernel_info(self, result, *, kernel_num=20, sigma=0.1):
        info = result.kernel_info
        if isinstance(info, dict):
            kernel_type = info["kernel"]
            actual_kernel_num = info["kernel_num"]
            actual_sigma = info["sigma"]
        else:
            kernel_type = info.kernel_type
            actual_kernel_num = info.kernel_num
            actual_sigma = info.sigma

        self.assertEqual(kernel_type, "Gaussian")
        self.assertEqual(actual_kernel_num, kernel_num)
        self.assertEqual(actual_sigma, sigma)

    def assert_kernel_weights(self, result, *, kernel_num=20):
        weights = getattr(result, "kernel_weights", getattr(result, "theta", None))
        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), kernel_num)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())

    def assert_density_ratio_for_new_input(self, result):
        new_x = np.linspace(0, 2, 11)
        density_ratio = result.compute_density_ratio(new_x)

        self.assertEqual(len(density_ratio), len(new_x))
        self.assertTrue(np.isfinite(density_ratio).all())
        self.assertTrue((density_ratio >= 0).all())

    def test_rulsif_fixed_parameter_contract(self):
        result = self.fit_rulsif()

        self.assertEqual(result.method, "RuLSIF")
        self.assertEqual(result.alpha, 0.1)
        self.assertEqual(getattr(result, "lambda_", getattr(result, "lambda", None)), 0.1)
        self.assert_kernel_info(result)
        self.assert_kernel_weights(result)
        self.assertTrue(callable(result.compute_density_ratio))

    def test_rulsif_computes_density_ratios_for_new_input_data(self):
        result = self.fit_rulsif()

        self.assert_density_ratio_for_new_input(result)

    def test_density_ratio_object_round_trips_through_pickle(self):
        result = self.fit_rulsif()
        new_x = np.linspace(0, 2, 11)
        expected_density_ratio = result.compute_density_ratio(new_x)

        restored = pickle.loads(pickle.dumps(result))
        actual_density_ratio = restored.compute_density_ratio(new_x)

        self.assertEqual(restored.method, result.method)
        self.assertEqual(restored.alpha, result.alpha)
        self.assertEqual(restored.lambda_, result.lambda_)
        np.testing.assert_allclose(actual_density_ratio, expected_density_ratio)

    def test_rulsif_multivariate_new_input_rows(self):
        rng = np.random.RandomState(314)
        x = rng.multivariate_normal(mean=[1, 1], cov=[[1.0 / 8.0, 0], [0, 1.0 / 8.0]], size=200)
        y = rng.multivariate_normal(mean=[1, 1], cov=[[1.0 / 2.0, 0], [0, 1.0 / 2.0]], size=200)
        new_x = np.column_stack((np.linspace(0, 2, 11), np.linspace(0, 2, 11)))

        np.random.seed(314)
        result = densratio(
            x,
            y,
            sigma_range=[0.1],
            lambda_range=[0.1],
            alpha=0.1,
            kernel_num=20,
            verbose=False,
        )
        density_ratio = result.compute_density_ratio(new_x)

        self.assertEqual(len(density_ratio), len(new_x))
        self.assertTrue(np.isfinite(density_ratio).all())
        self.assertTrue((density_ratio >= 0).all())

    def test_dimension_mismatch_raises_value_error(self):
        rng = np.random.RandomState(314)
        y_2d = rng.multivariate_normal(mean=[1, 1], cov=[[1.0 / 2.0, 0], [0, 2]], size=200)

        with self.assertRaises(ValueError):
            densratio(self.x, y_2d, verbose=False)

    @unittest.expectedFailure
    def test_r_public_exports_are_available(self):
        package = importlib.import_module("densratio")

        for name in ("densratio", "uLSIF", "RuLSIF", "KLIEP"):
            self.assertTrue(hasattr(package, name), name)

    @unittest.expectedFailure
    def test_densratio_accepts_method_argument_for_rulsif(self):
        result = densratio(
            self.x,
            self.y,
            method="RuLSIF",
            sigma_range=[0.1],
            lambda_range=[0.1],
            alpha=0.1,
            kernel_num=20,
            verbose=False,
        )

        self.assertEqual(result.method, "RuLSIF")
        self.assert_kernel_info(result)
        self.assert_kernel_weights(result)

    @unittest.expectedFailure
    def test_densratio_default_method_is_ulsif(self):
        result = densratio(
            self.x,
            self.y,
            sigma_range=[0.1],
            lambda_range=[1],
            kernel_num=20,
            verbose=False,
        )

        self.assertEqual(result.method, "uLSIF")
        self.assertEqual(getattr(result, "alpha", None), 0)
        self.assertEqual(getattr(result, "lambda_", getattr(result, "lambda", None)), 1)
        self.assert_kernel_info(result)
        self.assert_kernel_weights(result)
        self.assert_density_ratio_for_new_input(result)

    @unittest.expectedFailure
    def test_kliep_fixed_parameter_contract(self):
        package = importlib.import_module("densratio")

        np.random.seed(314)
        result = package.KLIEP(self.x, self.y, sigma=0.1, kernel_num=20, fold=5, verbose=False)

        self.assertEqual(result.method, "KLIEP")
        self.assertEqual(getattr(result, "fold", None), 5)
        self.assert_kernel_info(result)
        self.assert_kernel_weights(result)
        self.assert_density_ratio_for_new_input(result)


if __name__ == "__main__":
    unittest.main()
