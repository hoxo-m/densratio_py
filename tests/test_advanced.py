"""
Advanced test cases - with plots of computed alpha-relative density ratios.
"""

from .context import densratio
from scipy.stats import norm, cauchy, uniform, bernoulli
import numpy as np
import unittest
import matplotlib.pyplot as plt


class AdvancedTestSuite(unittest.TestCase):

    def test_plot_normalrv(self):
        def true_density_ratio(sample):
            return norm.pdf(sample, 0, 1./8) / (alpha * norm.pdf(sample, 0, 1./8) + (1 - alpha) * norm.pdf(sample, 0, 1./2))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./8)
        y = norm.rvs(size=200, loc=0, scale=1./2)
        alpha = 0
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_cauchyrv(self):
        def true_density_ratio(sample):
            return cauchy.pdf(sample, 0, 1./8) / (alpha * cauchy.pdf(sample, 0, 1./8) + (1 - alpha) * cauchy.pdf(sample, 0, 1./2))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = cauchy.rvs(size=200, loc=0, scale=1./8)
        y = cauchy.rvs(size=200, loc=0, scale=1./2)
        alpha = 0
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Cauchy Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_bimodalrv_1(self):
        def true_density_ratio(sample):
            weighted_pdf = (norm.pdf(sample, 0, 1./5) + norm.pdf(sample, 1, 1./5))/2
            return weighted_pdf / (alpha * weighted_pdf + (1 - alpha) * uniform.pdf(sample, loc=-10, scale=20))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        alpha = 0
        weight_toss = bernoulli.rvs(size=200, p=0.5)
        x = np.zeros((200, ))
        x[weight_toss == 1] = norm.rvs(size=np.where(weight_toss == 1)[0].shape[0], loc=1, scale=1./5)
        x[weight_toss == 0] = norm.rvs(size=np.where(weight_toss == 0)[0].shape[0], loc=0, scale=1./5)
        y = uniform.rvs(size=200, loc=-10, scale=20)
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Bimodal Density Ratio (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_bimodalrv_2(self):
        def true_density_ratio(sample):
            weighted_pdf = (norm.pdf(sample, 0, 1./5)*0.4 + norm.pdf(sample, 1, 1./5)*0.6)
            return weighted_pdf / (alpha * weighted_pdf + (1 - alpha) * uniform.pdf(sample, loc=-10, scale=20))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        alpha = 0
        weight_toss = bernoulli.rvs(size=200, p=0.6)
        x = np.zeros((200, ))
        x[weight_toss == 1] = norm.rvs(size=np.where(weight_toss == 1)[0].shape[0], loc=1, scale=1./5)
        x[weight_toss == 0] = norm.rvs(size=np.where(weight_toss == 0)[0].shape[0], loc=0, scale=1./5)
        y = uniform.rvs(size=200, loc=-10, scale=20)
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Bimodal Density Ratio (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_bimodalrv_3(self):
        def true_density_ratio(sample):
            weighted_pdf = (norm.pdf(sample, 0, 1. / 5) * 0.4 + norm.pdf(sample, 1, 1. / 5) * 0.6)
            return weighted_pdf / (alpha * weighted_pdf + (1 - alpha) * uniform.pdf(sample, loc=-10, scale=20))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        alpha = 0.2
        weight_toss = bernoulli.rvs(size=200, p=0.6)
        x = np.zeros((200,))
        x[weight_toss == 1] = norm.rvs(size=np.where(weight_toss == 1)[0].shape[0], loc=1, scale=1. / 5)
        x[weight_toss == 0] = norm.rvs(size=np.where(weight_toss == 0)[0].shape[0], loc=0, scale=1. / 5)
        y = uniform.rvs(size=200, loc=-10, scale=20)
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Bimodal Density Ratio (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_alpha1(self):
        def true_density_ratio(sample):
            return norm.pdf(sample, 0, 1./5) / (alpha * norm.pdf(sample, 0, 1./5) + (1 - alpha) * norm.pdf(sample, 0.5, 1./2))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./5, random_state=71)
        y = norm.rvs(size=200, loc=0.5, scale=1./2, random_state=71)
        alpha = 1
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_alpha2(self):
        def true_density_ratio(sample):
            return norm.pdf(sample, 0, 1./5) / (alpha * norm.pdf(sample, 0, 1./5) + (1 - alpha) * norm.pdf(sample, 0.5, 1./2))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./5, random_state=71)
        y = norm.rvs(size=200, loc=0.5, scale=1./2, random_state=71)
        alpha = 0.5
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_alpha3(self):
        def true_density_ratio(sample):
            return norm.pdf(sample, 0, 1./5) / (alpha * norm.pdf(sample, 0, 1./5) + (1 - alpha) * norm.pdf(sample, 0.5, 1./2))

        def estimated_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./5, random_state=71)
        y = norm.rvs(size=200, loc=0.5, scale=1./2, random_state=71)
        alpha = 0.2
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_density_ratio(sample_points), 'b-', label='True Density Ratio')
        plt.plot(sample_points, estimated_density_ratio(sample_points), 'r-', label='Estimated Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True


if __name__ == '__main__':
    unittest.main()
