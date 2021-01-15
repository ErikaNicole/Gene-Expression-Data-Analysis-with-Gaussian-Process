import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import classes.gp as gp
import classes.model_selection as ms

import warnings # Necessary to avoid certain common warnings from coming up. Won't affect inference.
warnings.filterwarnings('ignore')

my_path = os.path.dirname(os.path.realpath('__file__'))

class Test_Model_Selection(unittest.TestCase):

    def setUp(self):
        """
        The test cases chosen are 10 time series of non-oscillatory nature and 10 time series of oscillatory nature.
        In order to ensure accuracy in the model selection, we shall separate these into a observations set and
        into a control set, we construct the test cases using the GP class to create a prior trace from the OU
        and the OUosc processes respectively.
        """

        # Set up test case for whole test class:
        self.duration = 30 # hours
        self.num_of_observations = 50
        self.num_of_traces = 10

        # True parameters
        true_alpha = 0.2
        true_beta = 0.5         # only for oscillatory = True case
        true_variance = 1.0
        true_noise = 0.1

        # Prior Traces from the OU and OUosc processes respectively
        self.OU = gp.GP(true_alpha, 0, true_variance, true_noise, oscillatory = False)
        self.control_traces = self.OU.generate_prior_ou_trace(self.duration, self.num_of_observations, self.num_of_traces)
        self.OUosc = gp.GP(true_alpha, true_beta, true_variance, true_noise, oscillatory = True)
        self.observed_traces = self.OUosc.generate_prior_ou_trace(self.duration, self.num_of_observations, self.num_of_traces)

    def test_visualise_model_selection(self):

        # observed group model selection
        observed_group = ms.ModelSelection().model_selection(self.observed_traces[:,0], self.observed_traces[:, 1:-1],
                                                             number_of_synthetic_cells = 200)
        LLRs, optim_parameters, q_values, DistributionPlot, pi0Plot, QValuesPlot = observed_group

        # save plots
        DistributionPlot.savefig(os.path.join(my_path, "output/model_selection/test_LLR_histogram_plot_observed.pdf"))
        pi0Plot.savefig(os.path.join(my_path, "output/model_selection/test_pi0_plot_observed.pdf"))
        QValuesPlot.savefig(os.path.join(my_path, "output/model_selection/test_q_values_plot_observed.pdf"))

        # control group model selection
        control_group = ms.ModelSelection().model_selection(self.control_traces[:,0], self.control_traces[:, 1:-1],
                                                             number_of_synthetic_cells = 200)
        LLRs, optim_parameters, q_values, DistributionPlot, pi0Plot, QValuesPlot = control_group

        # save plots
        DistributionPlot.savefig(os.path.join(my_path, "output/model_selection/test_LLR_histogram_plot_control.pdf"))
        pi0Plot.savefig(os.path.join(my_path, "output/model_selection/test_pi0_plot_control.pdf"))
        QValuesPlot.savefig(os.path.join(my_path, "output/model_selection/test_q_values_plot_control.pdf"))













