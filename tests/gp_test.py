# Import

import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import classes.gp as gp

my_path = os.path.dirname(os.path.realpath('__file__'))

class Test_GP(unittest.TestCase):

    def setUp(self):

        # Set up test case for whole test class:
        self.duration = 100 # hours
        self.num_of_observations = 500

        # Oscillatory
        self.test_x = np.linspace(0, self.duration, self.num_of_observations)
        self.test_y_osc = np.sin(0.5*self.test_x) + np.random.normal(0, 0.1, len(self.test_x))
        self.observed_x = np.array([self.test_x[np.arange(0, 500, 10)]]).T
        self.observed_y_osc = self.test_y_osc[np.arange(0, 500, 10)]

        # Non-Oscillatory
        self.test_y = np.random.normal(0, 1, len(self.test_x))
        self.observed_y = self.test_y[np.arange(0, 500, 10)]

    def test_dimensionality_output(self):

        ou_osc_gauss_process = gp.GP(alpha = 0.04, beta = 0.5, variance = 1, noise = 0.1, oscillatory = True)

        # 1. cov_matrix_ou()
        cov_matrix = ou_osc_gauss_process.cov_matrix_ou(self.test_x, self.test_x, jitter = 0)
        self.assertEqual(np.shape(cov_matrix), tuple((len(self.test_x), len(self.test_x))))

        # 2. generate_prior_ou_trace()
        prior_trace = ou_osc_gauss_process.generate_prior_ou_trace(duration = self.duration,
                                                      number_of_observations = self.num_of_observations,
                                                      number_of_traces = 1)
        self.assertEqual(np.shape(prior_trace), tuple((len(self.test_x), 2))) # (number_of_observations, number_of_traces + 1)

        # 3. generate_predictor_and_posterior_ou_trace()
        posterior_trace = ou_osc_gauss_process.generate_predictor_and_posterior_ou_trace(self.observed_x, self.observed_y_osc,
                                                                                     self.duration, self.num_of_observations,
                                                                                     cholesky_decompose = True)
        self.assertEqual(np.shape(posterior_trace), tuple((len(self.test_x), 3))) # (number_of_observations, number_of_traces + 2)

    def test_visualise_gp(self):

        # Set up
        self.num_of_traces = 2

        # Oscillatory
        ou_osc_gauss_process = gp.GP(alpha = 0.04, beta = 0.5, variance = 1, noise = 0.1, oscillatory = True)

        prior_trace = ou_osc_gauss_process.generate_prior_ou_trace(self.duration,
                                                                   self.num_of_observations,
                                                                   self.num_of_traces)
        posterior_trace, post_confidence_bounds = ou_osc_gauss_process.generate_predictor_and_posterior_ou_trace(self.observed_x,
                                                                                         self.observed_y_osc,
                                                                                         self.duration,
                                                                                         self.num_of_observations,
                                                                                         self.num_of_traces,
                                                                                         confidence_bounds = True)

        prior_confidence_bounds = 1.96 * np.sqrt(np.diag(abs(ou_osc_gauss_process.cov_matrix_ou(self.test_x, self.test_x))))

        Fig, ax = plt.subplots(ncols=2, nrows=1, constrained_layout=True, sharey= True, figsize = (14, 10))
        #1st Plot
        ax[0].plot(self.test_x, np.zeros(len(self.test_x)), color = 'black', ls ='--')
        for i in range(1,self.num_of_traces + 1):
            ax[0].plot(self.test_x, prior_trace[:,i])
        ax[0].fill_between(self.test_x, + prior_confidence_bounds, - prior_confidence_bounds, alpha=0.2)
        ax[0].set(title ="Plot of OUosc Prior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        #2nd Plot
        for i in range(2, self.num_of_traces + 2):
            ax[1].plot(self.test_x, posterior_trace[:,i])
        ax[1].scatter(self.observed_x, self.observed_y_osc, color = 'red', alpha = 0.5, linewidths = 0.5)
        ax[1].plot(self.test_x, posterior_trace[:,1], color = 'black', ls ='--')
        ax[1].set(title ="Plot of OUosc Posterior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        ax[1].fill_between(self.test_x, posterior_trace[:,1] + post_confidence_bounds,
                           posterior_trace[:,1] - post_confidence_bounds, alpha=0.2)
        Fig.savefig(os.path.join(my_path, "output/test_oscillatory_gp_ou_fit.pdf"))

        # Non-Oscillatory
        ou_gauss_process = gp.GP(alpha = 0.04, beta = 0, variance = 1, noise = 0.1, oscillatory = False)

        prior_trace = ou_gauss_process.generate_prior_ou_trace(self.duration,
                                                               self.num_of_observations,
                                                               self.num_of_traces)
        posterior_trace, post_confidence_bounds = ou_gauss_process.generate_predictor_and_posterior_ou_trace(self.observed_x,
                                                                                         self.observed_y,
                                                                                         self.duration,
                                                                                         self.num_of_observations,
                                                                                         self.num_of_traces,
                                                                                         confidence_bounds = True)

        prior_confidence_bounds = 1.96 * np.sqrt(np.diag(abs(ou_gauss_process.cov_matrix_ou(self.test_x, self.test_x))))

        Fig, ax = plt.subplots(ncols=2, nrows=1, constrained_layout=True, sharey= True, figsize = (14, 10))
        #1st Plot
        ax[0].plot(self.test_x, np.zeros(len(self.test_x)), color = 'black', ls ='--')
        for i in range(1,self.num_of_traces + 1):
            ax[0].plot(self.test_x, prior_trace[:,i])
        ax[0].fill_between(self.test_x, + prior_confidence_bounds, - prior_confidence_bounds, alpha=0.2)
        ax[0].set(title ="Plot of OU Prior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        #2nd Plot
        for i in range(2, self.num_of_traces + 2):
            ax[1].plot(self.test_x, posterior_trace[:,i])
        ax[1].scatter(self.observed_x, self.observed_y, color = 'red', alpha = 0.5, linewidths = 0.5)
        ax[1].plot(self.test_x, posterior_trace[:,1], color = 'black', ls ='--')
        ax[1].set(title ="Plot of OU Posterior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        ax[1].fill_between(self.test_x, posterior_trace[:,1] + post_confidence_bounds,
                           posterior_trace[:,1] - post_confidence_bounds, alpha=0.2)
        Fig.savefig(os.path.join(my_path, "output/test_gp_ou_fit.pdf"))

        print("Look at output/test_gp_ou_fit.pdf and output/test_oscillatory_gp_ou_fit.pdf for visual test")

if __name__ == '__main__':
    unittest.main()








