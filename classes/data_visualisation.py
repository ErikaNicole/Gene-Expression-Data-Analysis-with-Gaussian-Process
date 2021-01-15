# A class built to visualise GP Regression results
import classes.gp as gp
import classes.data_prep as prep
import classes.optimisation as optimisation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits import mplot3d

sns.set_style("whitegrid")
my_path = os.getcwd()

# todo: include trace in the returns, to be able to access the data the plot came from.

class Visualisation_GP():

    def __init__(self, alpha, beta, variance, noise, oscillatory, observed_timepoints, observed_y, cholesky_decompose = True):
        """

        """
        self.alpha = alpha
        self.beta = beta
        self.variance = variance
        self.noise = noise
        self.oscillatory = oscillatory
        self.observed_timepoints = observed_timepoints
        self.observed_y = observed_y
        self.cholesky_decompose = cholesky_decompose
        self.GP = gp.GP(self.alpha, self.beta, self.variance, self.noise, self.oscillatory)
        self.prep = prep.Detrending(self.alpha, self.variance, self.noise)
        return

    def gp_ou_trace_plot(self, duration, number_of_observations, number_of_traces, prior = False):
        ''' Generate a plot for any number of samples from the Ornstein-Uhlenbeck (OU) process.
            Equivalent to plotting a number of samples from the Prior f* ~ N(0, K(X*, X*)) or Posterior f* ~ N(f*_mean, f*_covariance).
        For reference, see for example the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

        Parameters:
        -----------

        duration : float
            time duration/length of each trace that should be generated.

        number_of_observations : integer
            how many time points should be sampled from the process.

        number_of_traces: integer
            how many traces should be plotted.

        Returns:
        --------

        Fig : plot
            This figure has either of two plots, where it may contain the Prior Traces or the Posterior Traces, Predictor and Observed Points.
        '''

        timepoints = np.linspace(0.0,duration,number_of_observations)

        if prior:
                trace = self.GP.generate_prior_ou_trace(duration, number_of_observations, number_of_traces)

                confidence_bounds = 1.96 * np.sqrt(np.diag(abs(self.GP.cov_matrix_ou(timepoints, timepoints))))

                Fig = plt.figure('OU Prior Trace(s) Plot')
                plt.title('Plot of OU Prior Trace(s)')
                plt.ylabel('Expression')
                plt.xlabel('Time')
                for i in range(1,number_of_traces + 1):
                    plt.plot(timepoints, trace[:,i])
                plt.fill_between(timepoints, confidence_bounds, - confidence_bounds, alpha=0.1)
                plt.plot(timepoints, np.zeros(len(timepoints)), color = 'black', ls ='--') # Mean of Prior is just zeros.

        else:
                trace = self.GP.generate_predictor_and_posterior_ou_trace(self.observed_timepoints, self.observed_y, duration, number_of_observations, number_of_traces, self.cholesky_decompose, confidence_bounds = True)

                # Extract Traces, Predictor and Confidence Bounds from GP function
                confidence_bounds = trace[1]
                trace = trace[0]

                Fig = plt.figure('OU Posterior Trace(s) Plot')
                plt.title('Plot of OU Posterior Trace(s)')
                plt.ylabel('Expression')
                plt.xlabel('Time')
                for i in range(2,number_of_traces + 2):
                    plt.plot(timepoints, trace[:,i])
                plt.scatter(self.observed_timepoints, self.observed_y, color = 'red')
                plt.plot(timepoints, trace[:,1], color = 'black', ls ='--') # Predictor is column 1 of 'traces'.
                plt.fill_between(timepoints, trace[:,1] + confidence_bounds, trace[:,1] - confidence_bounds, alpha=0.2)

        return Fig, trace

    def gp_ou_trace_subplot(self, duration, number_of_observations, number_of_traces):
        ''' Generate a subplot for any number of samples from the Ornstein-Uhlenbeck (OU) process.
            Equivalent to plotting a number of samples from the Prior f* ~ N(0, K(X*, X*)) and Posterior f* ~ N(f*_mean, f*_covariance).
        For reference, see for example the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

        Parameters:
        -----------

        duration : float
            time duration/length of each trace that should be generated.

        number_of_observations : integer
            how many time points should be sampled from the process.

        number_of_traces: integer
            how many traces should be plotted.

        cholesky_decompose : boolean
            Default is True: constructs predictor and trains model using the cholesky decomposition of the Covariance Matrix.
            This is computationally more efficient and follows the Algorithm 2.1 as detailed in Rasmussen & Williams, 2006.

        Returns:
        --------

        Fig : plot
            This subplot has two plots, where the first plot contains Prior Traces,
            and the second contains the Posterior Traces, Predictor and Observed Points.
        '''

        timepoints = np.linspace(0.0,duration,number_of_observations)

        prior_trace = self.GP.generate_prior_ou_trace(duration, number_of_observations, number_of_traces)
        posterior_trace = self.GP.generate_predictor_and_posterior_ou_trace(self.observed_timepoints, self.observed_y, duration, number_of_observations, number_of_traces, self.cholesky_decompose, confidence_bounds = True)

        prior_confidence_bounds = 1.96 * np.sqrt(np.diag(abs(self.GP.cov_matrix_ou(timepoints, timepoints))))
        posterior_confidence_bounds = posterior_trace[1]
        posterior_trace = posterior_trace[0]

        Fig, ax = plt.subplots(ncols=2, nrows=1, constrained_layout=True, sharey= True)
        #1st Plot
        ax[0].plot(timepoints, np.zeros(len(timepoints)), color = 'black', ls ='--')
        for i in range(1,number_of_traces + 1):
            ax[0].plot(timepoints, prior_trace[:,i])
        ax[0].fill_between(timepoints, + prior_confidence_bounds, - prior_confidence_bounds, alpha=0.2)
        if self.oscillatory:
            ax[0].set(title ="Plot of OUosc Prior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        else:
            ax[0].set(title ="Plot of OU Prior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        #2nd Plot
        for i in range(2,number_of_traces + 2):
            ax[1].plot(timepoints, posterior_trace[:,i])
        ax[1].scatter(self.observed_timepoints, self.observed_y, color = 'red', alpha = 0.5, linewidths = 0.5)
        ax[1].plot(timepoints, posterior_trace[:,1], color = 'black', ls ='--')
        if self.oscillatory:
            ax[1].set(title ="Plot of OUosc Posterior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        else:
            ax[1].set(title ="Plot of OU Posterior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        ax[1].fill_between(timepoints, posterior_trace[:,1] + posterior_confidence_bounds, posterior_trace[:,1] - posterior_confidence_bounds, alpha=0.2)

        return Fig, prior_trace, posterior_trace

    def gp_ou_trace_3subplot(self, duration, number_of_observations, number_of_traces):
        ''' Generate a subplot for any number of samples from the Ornstein-Uhlenbeck (OU) process.
            Equivalent to plotting a number of samples from the Prior f* ~ N(0, K(X*, X*)) and Posterior f* ~ N(f*_mean, f*_covariance).
        For reference, see for example the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

        Parameters:
        -----------

        duration : float
            time duration/length of each trace that should be generated.

        number_of_observations : integer
            how many time points should be sampled from the process.

        number_of_traces: integer
            how many traces should be plotted.

        cholesky_decompose : boolean
            Default is True: constructs predictor and trains model using the cholesky decomposition of the Covariance Matrix.
            This is computationally more efficient and follows the Algorithm 2.1 as detailed in Rasmussen & Williams, 2006.

        Returns:
        --------

        Fig : plot
            This subplot has two plots, where the first plot contains Prior Traces,
            and the second contains the Posterior Traces, Predictor and Observed Points.
        '''

        timepoints = np.linspace(0.0,duration,number_of_observations)

        original_trace = self.observed_y
        prior_trace = self.GP.generate_prior_ou_trace(duration, number_of_observations, number_of_traces)
        posterior_trace = self.GP.generate_predictor_and_posterior_ou_trace(self.observed_timepoints, self.observed_y, duration, number_of_observations, number_of_traces, self.cholesky_decompose, confidence_bounds = True)

        prior_confidence_bounds = 1.96 * np.sqrt(np.diag(abs(self.GP.cov_matrix_ou(timepoints, timepoints))))
        posterior_confidence_bounds = posterior_trace[1]
        posterior_trace = posterior_trace[0]

        Fig, ax = plt.subplots(ncols=3, nrows=1, constrained_layout=True, sharey= True)
        #0st Plot
        ax[0].plot(self.observed_timepoints, original_trace)
        ax[0].set(title = 'Plot of Original Trace', ylabel = "Expression", xlabel = 'Time', ylim=[-3.5,3.5])

        #1st Plot
        ax[1].plot(timepoints, np.zeros(len(timepoints)), color = 'black', ls ='--')
        for i in range(1,number_of_traces + 1):
            ax[1].plot(timepoints, prior_trace[:,i])
        ax[1].fill_between(timepoints, + prior_confidence_bounds, - prior_confidence_bounds, alpha=0.2)
        if self.oscillatory:
            ax[1].set(title ="Plot of OUosc Prior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        else:
            ax[1].set(title ="Plot of OU Prior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])

        #2nd Plot
        for i in range(2,number_of_traces + 2):
            ax[2].plot(timepoints, posterior_trace[:,i])
        ax[2].scatter(self.observed_timepoints, self.observed_y, color = 'red', alpha = 0.5, linewidths = 0.5)
        ax[2].plot(timepoints, posterior_trace[:,1], color = 'black', ls ='--')
        if self.oscillatory:
            ax[2].set(title ="Plot of OUosc Posterior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        else:
            ax[2].set(title ="Plot of OU Posterior Trace(s)", ylabel = "Expression", xlabel = "Time", ylim=[-3.5,3.5])
        ax[2].fill_between(timepoints, posterior_trace[:,1] + posterior_confidence_bounds, posterior_trace[:,1] - posterior_confidence_bounds, alpha=0.2)

        return Fig, prior_trace, posterior_trace

    def gp_se_trace_plot(self, test_timepoints, number_of_traces, prior = False): #, duration = None, number_of_observations = None):
        ''' Generate a plot for any number of samples from the Ornstein-Uhlenbeck (OU) process.
            Equivalent to plotting a number of samples from the Prior f* ~ N(0, K(X*, X*)) or Posterior f* ~ N(f*_mean, f*_covariance).
        For reference, see for example the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

        Parameters:
        -----------

        duration : float
            time duration/length of each trace that should be generated.

        number_of_observations : integer
            how many time points should be sampled from the process.

        number_of_traces: integer
            how many traces should be plotted.

        Returns:
        --------

        Fig : plot
            This figure has either of two plots, where it may contain the Prior Traces or the Posterior Traces, Predictor and Observed Points.
        '''

        #if all(test_timepoints != None):
            #test_timepoints = test_timepoints
        #elif duration is not None and number_of_observations is not None:
            #test_timepoints = np.linspace(0.0,duration,number_of_observations)
        #else:
            #raise(ValueError("Inputs incorrect: choose between test_timepoints or a combination of duration and number_of_observations"))

        timepoints = test_timepoints

        if prior:
                trace = self.prep.generate_prior_SE_trace(test_timepoints, number_of_traces)

                confidence_bounds = 1.96 * np.sqrt(np.diag(abs(self.prep.cov_matrix_SE(timepoints, timepoints))))

                Fig = plt.figure('SE Prior Trace(s) Plot')
                plt.title('Plot of SE Prior Trace(s)')
                plt.ylabel('Expression')
                plt.xlabel('Time')
                for i in range(1,number_of_traces + 1):
                    plt.plot(timepoints, trace[:,i])
                plt.fill_between(timepoints, confidence_bounds, - confidence_bounds, alpha=0.1)
                plt.plot(timepoints, np.zeros(len(timepoints)), color = 'black', ls ='--') # Mean of Prior is just zeros.

        else:
                trace = self.prep.fit_SE(self.observed_timepoints, self.observed_y, test_timepoints, number_of_traces, self.cholesky_decompose)

                Fig = plt.figure('SE Posterior Trace(s) Plot')
                plt.title('Plot of SE Posterior Trace(s)')
                plt.ylabel('Expression')
                plt.xlabel('Time')
                for i in range(2,number_of_traces + 2):
                    plt.plot(timepoints, trace[:,i])
                plt.scatter(self.observed_timepoints, self.observed_y, color = 'red')
                plt.plot(timepoints, trace[:,1], color = 'black', ls ='--') # Predictor is column 1 of 'traces'.

        return Fig, trace

class Visualisation_Optimiser():

    def __init__(self, oscillatory, observed_timepoints, observed_y, optimizer_result, cholesky_decompose = True):
        """
        oscillatory: boolean
            Default is False: computes non-oscillatory Covariate_Matrix_OU.
            True constructs oscillatory Covariance_Matrix_OU.

        observed_timepoints: vector
            The vector of training inputs which have been observed.
            Should be same vector used in the optimizer

        observed_y: vector
            The vector of training inputs which have been observed.
            Should be same vector used in the optimizer

        optimizer_result: OptimizeResult, or, tuple
            If you had run the optimizer function 'optimizing_neg_marginal_loglikelihood' with callback = True, it would return a tuple.
            If you had run the optimizer function 'optimizing_neg_marginal_loglikelihood' with callback = False, it would return an OptimizeResult.

            This function can handle either case.
            'optimizing_neg_marginal_loglikelihood' is found within Optimisation() class in optimisation.py

        """
        self.oscillatory = oscillatory
        self.observed_timepoints = observed_timepoints
        self.observed_y = observed_y
        self.optimizer_result = optimizer_result
        self.cholesky_decompose = cholesky_decompose
        self.optim = optimisation.Optimisation(oscillatory, observed_timepoints, observed_y)
        return

    def plotting_hyperparameters_densities_2d(self):
        """
        Function for plotting Hyper Parameters densities.
        This is achieved by utilizing the optimizer's result to their fullest.
        Picking out the optimized Hyper Parameters and fixing those, in turn, to plot the densities for a range of possible values.
        This could shed some light about whether we have hit a local maxima or a global one.

        Returns:
            Fig : plot
            Subplot of densities of individual Hyper Parameters, keeping all others constant.
        '''
        """
        if len(self.optimizer_result) == 2:
            results = self.optimizer_result[0]
        else:
            results = self.optimizer_result

        xline = np.linspace(0.0, 5, 100)
        optimized_alpha, optimized_beta, optimized_variance, optimized_noise = results.x[0], results.x[1], results.x[2], results.x[3]
        maximum_marginal_llik = -results.fun

        # 1. Create Grid of Subplots
        Fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
        Fig.suptitle('Density Plots of Optimized Hyper Parameters with respect to Log Marginal Likelihood', fontsize=12)

        # 2. Specify each Ax Contents
        #1st Plot
        yline = np.zeros(len(xline))
        for index, x in enumerate(xline):
            yline[index] = -self.optim.neg_marginal_loglikelihood([x, optimized_beta, optimized_variance, optimized_noise], cholesky_decompose = self.cholesky_decompose)

        ax[0,0].plot(xline, yline)
        ax[0,0].plot(optimized_alpha, maximum_marginal_llik, color = 'red', marker = '+', ms = 8)
        ax[0,0].set(title ="Density Plot for Alpha", ylabel = "Log Marginal Likelihood", xlabel = "Alpha")

        #2nd Plot
        if self.oscillatory:
            yline = np.zeros(len(xline))
            for index, x in enumerate(xline):
                yline[index] = -self.optim.neg_marginal_loglikelihood([optimized_alpha, x, optimized_variance, optimized_noise], cholesky_decompose = self.cholesky_decompose)

            ax[0,1].plot(xline, yline, color = 'orange')
            ax[0,1].plot(optimized_beta, maximum_marginal_llik, color = 'red', marker = '+', ms = 8)
            ax[0,1].set(title ="Density Plot for Beta", ylabel = "Log Marginal Likelihood", xlabel = "Beta")
        else:
            ax[0,1].set(title ="Beta is not involved in the Non-Oscillatory Process", ylabel = "Log Marginal Likelihood", xlabel = "Beta")
            ax[0,1].annotate("NA", xy = (0.45,0.5))

        #3rd Plot
        yline = np.zeros(len(xline))
        for index, x in enumerate(xline):
            yline[index] = -self.optim.neg_marginal_loglikelihood([optimized_alpha, optimized_beta, x, optimized_noise], cholesky_decompose = self.cholesky_decompose)

        ax[1,0].plot(xline, yline, color = 'orange')
        ax[1,0].plot(optimized_variance, maximum_marginal_llik, color = 'red', marker = '+', ms = 8)
        ax[1,0].set(title = "Density Plot for Variance",  ylabel = "Log Marginal Likelihood", xlabel = "Variance")

        #4th Plot
        yline = np.zeros(len(xline))
        for index, x in enumerate(xline):
            yline[index] = -self.optim.neg_marginal_loglikelihood([optimized_alpha, optimized_beta, optimized_variance, x], cholesky_decompose = self.cholesky_decompose)

        ax[1,1].plot(xline, yline, color = 'orange')
        ax[1,1].plot(optimized_noise, maximum_marginal_llik, color = 'red', marker = '+', ms = 8)
        ax[1,1].set(title = "Density Plot for Noise", ylabel = "Log Marginal Likelihood", xlabel = "Noise")

        return Fig

    def plotting_hyperparameters_densities_3d(self, range, hyperparameters):
        """
        Function for plotting Hyper Parameters densities as 3D Surfaces.
        This is achieved by utilizing the optimizer's result to their fullest.
        Picking out the optimized Hyper Parameters and fixing those, in turn, to plot the densities for a range of possible values.
        This could shed some light about whether we have hit a local maxima or a global one.

        Parameters:
        -----------

        range: list
            List containing the range for the hyperparameter to be plotted of the form [a, b].
            Note that the range in certain cases might have to be strictly larger than zero or the optimizer will return an error.

        hyperparameters: list
            List of the form ['hyperparameter1', 'hyperparameter2'] containing the two hyperparameters to be plotted
            in the x and y axes respectively. The possible combinations are:

            ['alpha', 'beta']
            ['alpha', 'variance']
            ['alpha', 'noise']
            ['beta', 'variance']
            ['beta', 'noise']
            ['variance', 'noise']

        Returns:
            Fig : plot
            Subplot of densities of individual Hyper Parameters, keeping all others constant.
        '''
        """
        if len(self.optimizer_result) == 2:
            results = self.optimizer_result[0]
        else:
            results = self.optimizer_result

        if hyperparameters[0] == 'beta' or hyperparameters[1] == 'beta' and self.oscillatory == False:
            raise(ValueError("You cannot plot beta if oscillatory boolean is set as False"))

        optimized_alpha, optimized_beta, optimized_variance, optimized_noise = results.x[0], results.x[1], results.x[2], results.x[3]

        x = np.linspace(range[0], range[1], 10)
        y = np.linspace(range[0], range[1], 10)
        X, Y = np.meshgrid(x, y)

        if hyperparameters == ['alpha', 'variance']:
            Z = np.zeros((len(X), len(X)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    Z[i,j] = -self.optim.neg_marginal_loglikelihood([xi, optimized_beta, yj, optimized_noise], cholesky_decompose = self.cholesky_decompose)
        elif hyperparameters == ['alpha', 'beta']:
            Z = np.zeros((len(X), len(X)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    Z[i,j] = -self.optim.neg_marginal_loglikelihood([xi, yj, optimized_variance, optimized_noise], cholesky_decompose = self.cholesky_decompose)
        elif hyperparameters == ['alpha', 'noise']:
            Z = np.zeros((len(X), len(X)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    Z[i,j] = -self.optim.neg_marginal_loglikelihood([xi, optimized_beta, optimized_variance, yj], cholesky_decompose = self.cholesky_decompose)
        elif hyperparameters == ['beta', 'variance']:
            Z = np.zeros((len(X), len(X)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    Z[i,j] = -self.optim.neg_marginal_loglikelihood([optimized_alpha, xi, yj, optimized_noise], cholesky_decompose = self.cholesky_decompose)
        elif hyperparameters == ['beta', 'noise']:
            Z = np.zeros((len(X), len(X)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    Z[i,j] = -self.optim.neg_marginal_loglikelihood([optimized_alpha, xi, optimized_variance, yj], cholesky_decompose = self.cholesky_decompose)
        elif hyperparameters == ['variance', 'noise']:
            Z = np.zeros((len(X), len(X)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    Z[i,j] = -self.optim.neg_marginal_loglikelihood([optimized_alpha, optimized_beta, xi, yj], cholesky_decompose = self.cholesky_decompose)
        else:
            raise(ValueError("The given combination of hyperparameters is not valid, pick one of the options listed"))

        Fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.set_xlabel(hyperparameters[0])
        ax.set_ylabel(hyperparameters[1])
        ax.set_zlabel('Log Marginal Likelihood')
        ax.view_init(30, 35)

        return Fig

class Visualisation_ModelSelection():

    def __init__(self):
        return

    def LLR_distribution_plot(self, observed_LLRs, synthetic_LLRs):
        """
        Function for plotting the approximate distribution of LLRs for a population of non-oscillating cells.
        The subplots of control and observed cells LLRs is to make an initial estimation as to whether the gene seems to be
        behaving in an oscillatory or non-oscillatory manner.

        From the Model Selection function approximation_of_LLRs we can easily obtain relevant LLRs.

        When examining the histograms consider:
        LLRs distributions of observed cells which behave similarly as

        Parameters:
        --------
        observed_LLRs: ndarray

        synthetic_LLRs: ndarray

        Returns:
        -------

        Plot: Fig
            Plot of the approximate distribution of the LLRs for a population of non-oscillating cells to compare with
            results of control cells and observed cells.
        """

        Plot, ax = plt.subplots(ncols=2, nrows=1, constrained_layout=True,  figsize = (12,8))
        # 1st plot of observed oscillating cells
        ax[0].hist(observed_LLRs)
        ax[0].set(title = "Observed LLRs Distribution", ylabel = "Frequency", xlabel = "LLR")
        # 2rd plot of null hypothesis - population of non-oscillating cells
        ax[1].hist(synthetic_LLRs)
        ax[1].set(title = "Synthetic LLRs Distribution", ylabel = "Frequency", xlabel = "LLR")

        return Plot

    def LLR_distribution_plot_with_control(self, control_LLRs, observed_LLRs, synthetic_LLRs):
        """
        Function for plotting the approximate distribution of LLRs for a population of non-oscillating cells.
        The subplots of control and observed cells LLRs is to make an initial estimation as to whether the gene seems to be
        behaving in an oscillatory or non-oscillatory manner.

        From the Model Selection function approximation_of_LLRs we can easily obtain relevant LLRs.

        When examining the histograms consider:
        LLRs distributions of observed cells which behave similarly as

        Parameters:
        --------

        control_LLRs: ndarray

        observed_LLRs: ndarray

        synthetic_LLRs: ndarray

        Returns:
        -------

        Plot: Fig
            Plot of the approximate distribution of the LLRs for a population of non-oscillating cells to compare with
            results of control cells and observed cells.
        """

        Plot1, ax = plt.subplots(ncols=3, nrows=1, constrained_layout=True, sharex = True)
        # max_value = max(max(observed_LLRs), max(control_LLRs))
        # min_value = min(min(observed_LLRs), min(control_LLRs))
        # binwidth = 0.5

        # 1st plot of observed oscillating cells
        ax[0].hist(observed_LLRs)
        #ax[0].hist(observed_LLRs, bins = np.arange(min_value, max_value + binwidth, binwidth))
        ax[0].set(title = "Observed LLRs Distribution", ylabel = "Frequency", xlabel = "LLR")
        # 2nd plot of control non-oscillating cells
        ax[1].hist(control_LLRs)
        ax[1].set(title = "Control LLRs Distribution", ylabel = "Frequency", xlabel = "LLR")
        # 3rd plot of null hypothesis - population of non-oscillating cells
        ax[2].hist(synthetic_LLRs)
        ax[2].set(title = "Synthetic LLRs Distribution", ylabel = "Frequency", xlabel = "LLR")
        plt.show()

        # Second plot shows densities overlapping
        # Plot2 = plt.figure("Densities of Control (blue), Observed (green), and Synthetic (red)")
        # plt.title("Densities of Control (blue), Observed (green), and Synthetic (red)")
        # kwargs = dict(histtype='stepfilled', alpha=0.3, bins=np.arange(min_value, max_value + binwidth, binwidth))
        # plt.hist(control_LLRs, **kwargs)
        # plt.hist(observed_LLRs, **kwargs)
        # plt.hist(synthetic_LLRs, **kwargs)
        # plt.show()

        return Plot1

    def q_values_plot(self, LLRs, q_est, control_q_value):
        """
        Q-values depends on LLR of data we can plot the following.

        At an FDR of 5%, ?/? cells would pass the LLR threshold and are classified as oscillatory.
        If the FDR is made less stringent at 10%, the pass rate could change significantly. It can easily be seen from the plot.

        By controlling at a cutoff of 5% we ensure that the rate of significant features which are truly null is more
        realistic. For example a q value of 0.0318 is the expected proportion of false positives incurred if we called the gene signficant.

        Returns:
        ------

        Fig: Fig
            Plot showing q-value distribution of observed cells and control cells.

        """

        Fig = plt.figure('q-values plot',  figsize = (10,6))
        plt.ylabel("Q-Values")
        plt.xlabel("Threshold")
        plt.scatter(np.sort(LLRs), q_est)
        plt.axhline(control_q_value, color = 'red')
        plt.ylim([-0.1,1])
        plt.title("Q-Value Distribution depending on different threshold values")

        return Fig

class Visualisation_DataPrep():

    def __init__(self, detrending_alpha = 0.0001, detrending_variance = 1.0, detrending_noise = 0.0001):
        self.alpha = detrending_alpha
        self.variance = detrending_variance
        self.noise = detrending_noise
        self.prep = prep.Detrending(alpha = self.alpha, variance = self.variance, noise = self.noise)
        return

    def detrending_plot(self, time, normalised_cell_data):
        """
        Should be used for one example cell only. Otherwise will end up being too busy.
        This plot will show the original signal, the fitted trend and the detrended signal.

        Parameters:
        ----------

        normalised_cell_data: ndarray or list
            This data can be retrieved directly from the other data prep functions.

        Returns:
        ----------

        Fig: Fig
            Plot of Detrended Data, Original Signal and GP with SE covariance function fit.
        """

        if isinstance(normalised_cell_data, np.ndarray):
            detrending = self.prep.detrend_data(observed_timepoints = time, data_to_detrend = normalised_cell_data)
        else:
            print("Function not ready to take list input")
            # detrending = self.prep.detrend_data_from_list(observed_timepoints = time, data_to_detrend = normalised_cell_data)

        detrended_data = detrending[0]
        trendfit_x = detrending[1]
        trendfit_y = detrending[2]

        Fig = plt.figure("Detrended Data")
        plt.xlabel("Time (Hours)")
        plt.ylabel("Gene Expression")
        plt.plot(time, detrended_data)
        plt.plot(time, normalised_cell_data)
        plt.plot(trendfit_x, trendfit_y, ls = '--')

        return Fig

