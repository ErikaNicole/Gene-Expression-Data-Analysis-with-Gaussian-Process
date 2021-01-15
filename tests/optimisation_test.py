import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import classes.gp as gp
import classes.optimisation as optim

import warnings # Necessary to avoid certain common warnings from coming up. Won't affect inference.
warnings.filterwarnings('ignore')

my_path = os.path.dirname(os.path.realpath('__file__'))

class Test_Optimisation(unittest.TestCase):

    def setUp(self):
        """
        The test cases chosen are one of non-oscillatory nature and one of oscillatory nature.
        In order to ensure the optimisation accuracy in choosing the most likely set of parameters, we construct the
        test cases using the GP class to create a prior trace from the OU and the OUosc processes respectively.
        """

        # Set up test case for whole test class:
        self.duration = 50 # hours
        self.num_of_observations = 100

        # True parameters
        true_alpha = 0.7
        true_beta = 0.5         # only for oscillatory = True case
        true_variance = 1.0
        true_noise = 0.1

        # Prior Traces from the OU and OUosc processes respectively
        self.OU = gp.GP(true_alpha, 0, true_variance, true_noise, oscillatory = False)
        self.observed_trace = self.OU.generate_prior_ou_trace(self.duration, self.num_of_observations, 1)
        self.OUosc = gp.GP(true_alpha, true_beta, true_variance, true_noise, oscillatory = True)
        self.observed_trace_osc = self.OUosc.generate_prior_ou_trace(self.duration, self.num_of_observations, 1)

    def test_dimensionality_output(self):

        OU_optim = optim.Optimisation(False, self.observed_trace[:,0], self.observed_trace[:, 1])

        # 1. neg_marginal_loglikelihood()
        neg_marg_llik_value = OU_optim.neg_marginal_loglikelihood(theta = [0.1, 0.0, 0.1, 0.1])
        self.assertTrue(isinstance(neg_marg_llik_value, float))

        # 2. optimizing_neg_marginal_loglikelihood()
        optim_param = OU_optim.optimizing_neg_marginal_loglikelihood(start_values = [0.5, 0.0, 0.5, 0.5])
        self.assertEqual(len(optim_param), 9)
        self.assertEqual(len(optim_param.x), 4) # four parameters estimated (even if non osc, but beta = 0)

    def test_visualise_optimised_trace(self):

        # - - - -
        # Optimise and obtain new traces using optimised parameters
        OU_optim = optim.Optimisation(False, self.observed_trace[:,0], self.observed_trace[:, 1])
        OUosc_optim = optim.Optimisation(True, self.observed_trace_osc[:,0], self.observed_trace_osc[:, 1])

        optim_OU_param = OU_optim.optimizing_neg_marginal_loglikelihood(start_values = [0.5, 0.0, 0.5, 0.5])
        optim_OUosc_param = OUosc_optim.optimizing_neg_marginal_loglikelihood(start_values = [0.5, 0.5, 0.5, 0.5])

        optim_OU_trace = gp.GP(optim_OU_param.x[0], optim_OU_param.x[1], optim_OU_param.x[2], optim_OU_param.x[3],
                               oscillatory = False)
        optim_OU_post_trace, post_OU_confidence_bounds = optim_OU_trace.generate_predictor_and_posterior_ou_trace(self.observed_trace[:,0],
                                                                                       self.observed_trace[:,1],
                                                                                       duration = 50,
                                                                                       confidence_bounds = True)
        optim_OUosc_trace = gp.GP(optim_OUosc_param.x[0], optim_OUosc_param.x[1], optim_OUosc_param.x[2],
                                  optim_OUosc_param.x[3], oscillatory = True)
        optim_OUosc_post_trace, post_OUosc_confidence_bounds = optim_OUosc_trace.generate_predictor_and_posterior_ou_trace(self.observed_trace_osc[:,0],
                                                                                            self.observed_trace_osc[:,1],
                                                                                            duration = 50,
                                                                                            confidence_bounds = True)

        # Plot set up

        Fig, ax = plt.subplots(ncols = 2, nrows = 1, constrained_layout = True, sharey = True, figsize = (14, 10))
        Fig.suptitle('OU and OUosc Optimisation of Traces with param = (alpha = 0.7, beta = 0.5, variance = 1.0, noise = 0.1)')

        # - - - -

        # 1st Plot
        # Non-Oscillatory

        ax[0].plot(self.observed_trace[:, 0], self.observed_trace[:, 1], color = 'red')  # real trace
        ax[0].scatter(self.observed_trace[:, 0], self.observed_trace[:,1],
                      color = 'red', alpha = 0.5, linewidths = 0.5)                      # real observed points
        ax[0].plot(optim_OU_post_trace[:,0], optim_OU_post_trace[:,1], color = 'black', ls ='--')
        ax[0].fill_between(optim_OU_post_trace[:,0], optim_OU_post_trace[:,1] + post_OU_confidence_bounds,
                   optim_OU_post_trace[:,1] - post_OU_confidence_bounds, alpha=0.2)
        ax[0].set(title = "OU True Trace (red) and Fitted Trace (black)",
                  ylabel = "Expression", xlabel = "Time", ylim = [-3.5,3.5])
        ax[0].text(0, 3, "optimised parameters : " + str("{:.2f}".format(optim_OU_param.x[0])) + ' , ' +
                   str("{:.2f}".format(optim_OU_param.x[2])) + ' , ' +
                   str("{:.2f}".format(optim_OU_param.x[3])), fontsize = 10, color='grey')

        # - - - -

        # 2nd Plot
        # Oscillatory

        ax[1].plot(self.observed_trace_osc[:, 0], self.observed_trace_osc[:, 1],        # real trace
                   color = 'red')
        ax[1].scatter(self.observed_trace_osc[:, 0], self.observed_trace_osc[:, 1],
              color = 'red', alpha = 0.5, linewidths = 0.5)                             # real observed points
        ax[1].plot(optim_OUosc_post_trace[:,0], optim_OUosc_post_trace[:,1], color = 'black', ls ='--')
        ax[1].fill_between(optim_OUosc_post_trace[:,0], optim_OUosc_post_trace[:,1] + post_OUosc_confidence_bounds,
                   optim_OUosc_post_trace[:,1] - post_OUosc_confidence_bounds, alpha=0.2)
        ax[1].set(title = "OUosc True Trace (red) and Fitted Trace (black)",
                  ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        ax[1].text(0, 3, "optimised parameters : " + str("{:.2f}".format(optim_OUosc_param.x[0])) + ' , ' +
                   str("{:.2f}".format(optim_OUosc_param.x[1])) + ' , ' +
                   str("{:.2f}".format(optim_OUosc_param.x[2])) + ' , ' +
                   str("{:.2f}".format(optim_OUosc_param.x[3])), fontsize = 10, color='grey')

        Fig.savefig(os.path.join(my_path, "output/test_optimisation_fitting_traces.pdf"))

        print("Look at output/test_optimisation_fitting_traces.pdf for visual test")

if __name__ == '__main__':
    unittest.main()
