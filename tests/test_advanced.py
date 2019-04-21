"""
Advanced test cases - with plots of computed alpha-relative density ratios.
"""

from .context import densratio
from scipy.stats import norm, cauchy, uniform, bernoulli, multivariate_normal
import numpy as np
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AdvancedTestSuite(unittest.TestCase):

    def test_plot_normalrv_1(self):
        def true_alpha_density_ratio(sample):
            return norm.pdf(sample, 0, 1./8) / (alpha * norm.pdf(sample, 0, 1./8) + (1 - alpha) * norm.pdf(sample, 0, 1./2))

        def estimated_alpha_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./8)
        y = norm.rvs(size=200, loc=0, scale=1./2)
        alpha = 0.1
        densratio_obj = densratio(x, y, alpha=alpha)
        print(densratio_obj)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_normalrv_2(self):
        def true_alpha_density_ratio(sample):
            return norm.pdf(sample, 0, 1./8) / (alpha * norm.pdf(sample, 0, 1./8) + (1 - alpha) * norm.pdf(sample, 0, 1./2))

        def estimated_alpha_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./8)
        y = norm.rvs(size=200, loc=0, scale=1./2)
        alpha = 0.1
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_cauchyrv(self):
        def true_alpha_density_ratio(sample):
            return cauchy.pdf(sample, 0, 1./8) / (alpha * cauchy.pdf(sample, 0, 1./8) + (1 - alpha) * cauchy.pdf(sample, 0, 1./2))

        def estimated_alpha_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = cauchy.rvs(size=200, loc=0, scale=1./8)
        y = cauchy.rvs(size=200, loc=0, scale=1./2)
        alpha = 0
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Cauchy Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_bimodalrv_1(self):
        def true_alpha_density_ratio(sample):
            weighted_pdf = (norm.pdf(sample, 0, 1./5) + norm.pdf(sample, 1, 1./5))/2
            return weighted_pdf / (alpha * weighted_pdf + (1 - alpha) * uniform.pdf(sample, loc=-10, scale=20))

        def estimated_alpha_density_ratio(sample):
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
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Bimodal Density Ratio (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_bimodalrv_2(self):
        def true_alpha_density_ratio(sample):
            weighted_pdf = (norm.pdf(sample, 0, 1./5)*0.4 + norm.pdf(sample, 1, 1./5)*0.6)
            return weighted_pdf / (alpha * weighted_pdf + (1 - alpha) * uniform.pdf(sample, loc=-10, scale=20))

        def estimated_alpha_density_ratio(sample):
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
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Bimodal Density Ratio (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_bimodalrv_3(self):
        def true_alpha_density_ratio(sample):
            weighted_pdf = (norm.pdf(sample, 0, 1. / 5) * 0.4 + norm.pdf(sample, 1, 1. / 5) * 0.6)
            return weighted_pdf / (alpha * weighted_pdf + (1 - alpha) * uniform.pdf(sample, loc=-10, scale=20))

        def estimated_alpha_density_ratio(sample):
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
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Bimodal Density Ratio (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_alpha1(self):
        def true_alpha_density_ratio(sample):
            return norm.pdf(sample, 0, 1./5) / (alpha * norm.pdf(sample, 0, 1./5) + (1 - alpha) * norm.pdf(sample, 0.5, 1./2))

        def estimated_alpha_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./5, random_state=71)
        y = norm.rvs(size=200, loc=0.5, scale=1./2, random_state=71)
        alpha = 1
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_alpha2(self):
        def true_alpha_density_ratio(sample):
            return norm.pdf(sample, 0, 1./5) / (alpha * norm.pdf(sample, 0, 1./5) + (1 - alpha) * norm.pdf(sample, 0.5, 1./2))

        def estimated_alpha_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./5, random_state=71)
        y = norm.rvs(size=200, loc=0.5, scale=1./2, random_state=71)
        alpha = 0.5
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test_plot_alpha3(self):
        def true_alpha_density_ratio(sample):
            return norm.pdf(sample, 0, 1./5) / (alpha * norm.pdf(sample, 0, 1./5) + (1 - alpha) * norm.pdf(sample, 0.5, 1./2))

        def estimated_alpha_density_ratio(sample):
            return densratio_obj.compute_density_ratio(sample)

        np.random.seed(1)
        x = norm.rvs(size=200, loc=0, scale=1./5, random_state=71)
        y = norm.rvs(size=200, loc=0.5, scale=1./2, random_state=71)
        alpha = 0.2
        densratio_obj = densratio(x, y, alpha=alpha)
        sample_points = np.linspace(-1, 3, 400)
        plt.plot(sample_points, true_alpha_density_ratio(sample_points), 'b-', label='True Alpha-Relative Density Ratio')
        plt.plot(sample_points, estimated_alpha_density_ratio(sample_points), 'r-', label='Estimated Alpha-Relative Density Ratio')
        plt.title("Alpha-Relative Density Ratio - Normal Random Variables (alpha={:03.2f})".format(alpha))
        plt.legend()
        plt.show()
        assert True

    def test2d(self):
        def true_alpha_density_ratio(x):
            return multivariate_normal.pdf(x, [1., 1.], [[1. / 8, 0], [0, 1. / 8]]) / \
                   (alpha * multivariate_normal.pdf(x, [1., 1.], [[1. / 8, 0], [0, 1. / 8]]) + (1 - alpha) * multivariate_normal.pdf(x, [1., 1.], [[1. / 2, 0], [0, 1. / 2]]))

        def estimated_alpha_density_ratio(x):
            return densratio_obj.compute_density_ratio(x)

        x = multivariate_normal.rvs(size=3000, mean=[1, 1], cov=[[1. / 8, 0], [0, 1. / 8]])
        y = multivariate_normal.rvs(size=3000, mean=[1, 1], cov=[[1. / 2, 0], [0, 1. / 2]])
        alpha = 0
        densratio_obj = densratio(x, y, alpha=alpha, sigma_range=[0.1, 0.3, 0.5, 0.7, 1], lambda_range=[0.01, 0.02, 0.03, 0.04, 0.05])
        
        print(densratio_obj)
        
        range_ = np.linspace(0, 2, 200)
        grid = np.concatenate(np.dstack(np.meshgrid(range_, range_)))
        levels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5]

        plt.subplot(1, 2, 1)
        plt.contourf(range_, range_, true_alpha_density_ratio(grid).reshape(200, 200), levels)
        plt.colorbar()
        plt.title("True Alpha-Relative Density Ratio")
        plt.subplot(1, 2, 2)
        plt.contourf(range_, range_, estimated_alpha_density_ratio(grid).reshape(200, 200), levels)
        plt.colorbar()
        plt.title("Estimated Alpha-Relative Density Ratio")
        plt.show()


if __name__ == '__main__':
    unittest.main()
