# Import

import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import classes.data_prep as data_prep

my_path = os.path.dirname(os.path.realpath('__file__'))

class Test_data_prep(unittest.TestCase):

    def setUp(self):

        # Set up test case for whole test class
        self.test_x = np.linspace(0, 100, 500)
        self.test_y = np.sin(0.5*self.test_x) + np.random.normal(0, 0.5, len(self.test_x))
        self.observed_x = np.array([self.test_x[np.arange(0, 500, 10)]]).T
        self.observed_y = np.array([self.test_y[np.arange(0, 500, 10)]]).T

        # Introduce three different trends on the same observed data
        self.data = np.hstack((self.observed_y, self.observed_y))
        self.data = np.hstack((self.data, self.observed_y))
        for i in range(50):
            self.data[i,0] = self.data[i,0] + i*0.15
            self.data[i,1] = self.data[i,1] + np.sin(0.2*self.observed_x[i, 0])
            self.data[i,2] = self.data[i,2] + np.exp(0.01*self.observed_x[i, 0])

    def test_dimensionality_output(self):
        """
        Aims to check the dimensionality of all outputs and ensuring they are the same as per the function descriptions
        """

        detrending = data_prep.Detrending(alpha = 0.001,
                                          variance = 0.001,
                                          noise = 0.001)

        matrix = detrending.cov_matrix_SE(self.test_x, self.test_x)
        prior_trace = detrending.generate_prior_SE_trace(self.test_x, 1)

        self.assertEqual(np.shape(matrix), tuple((len(self.test_x), len(self.test_x))))
        self.assertEqual(np.shape(prior_trace), tuple((len(self.test_x), 2))) # (number_of_observations, number_of_traces + 1)
        # self.assertEqual(prior_trace[:,0], self.test_x)

        optim = detrending.optimise_SE_trace(0.001, 0.001, 0.001, self.test_x, self.test_y)

        self.assertTrue(len(optim) == 3) # three parameters optimised

        posterior_trace = detrending.fit_SE(self.observed_x, self.observed_y, self.test_x, 1, True)

        self.assertEqual(np.shape(posterior_trace), tuple((len(self.test_x), 3))) # (number_of_observations, number_of_traces + 2)

        detrended_data, test_timepoints, fits = detrending.detrend_data(self.observed_x, self.data, True)

        self.assertTrue(np.shape(detrended_data) == np.shape(self.data))
        self.assertTrue(np.shape(fits) == tuple((len(test_timepoints), np.shape(detrended_data)[1])))

    def test_visualise_detrending(self):
        """
        Aims to produce plot of detrending process on the same synthetic time series which has been modified with three
        different trends.
        Optimal outcome : all three detrended time series should be as close as possible
        """
        detrending = data_prep.Detrending(alpha = 0.001,
                                          variance = 0.001,
                                          noise = 0.001)

        detrended_data, test_timepoints, fits = detrending.detrend_data(self.observed_x, self.data, True)

        fig = plt.figure("detrending test", figsize = (10,6))
        plt.plot(self.observed_x, self.observed_y, color = 'red', label = "True Detrended Signal", ls ='--')
        for i in range(3):
            optim = detrending.optimise_SE_trace(0.001, 0.001, 0.001, self.observed_x, self.data[:,i])
            print("optimised parameters for detrending used were", optim)

            plt.plot(self.observed_x, self.data[:,i], label = "Observed Data", color = "darkblue") # observed data
            plt.plot(self.observed_x, detrended_data[:,i], label = "Detrended Data", color = "lightblue") # detrended data
            plt.plot(test_timepoints, fits[:,i], color = 'black', ls ='--', label = "Fitted SE") # predictor

        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.title('Detrending Process')
        fig.savefig(os.path.join(my_path, "output/detrending_test.pdf"))

if __name__ == '__main__':
    unittest.main()

