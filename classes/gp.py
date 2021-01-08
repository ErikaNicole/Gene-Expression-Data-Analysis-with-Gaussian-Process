import numpy as np
from numpy.linalg import cholesky, det, lstsq, inv
from scipy.linalg import cholesky, cho_solve

class GP():

    def __init__(self, alpha, beta, variance, noise, oscillatory):
        """
        Instance Variables
        alpha: float
            The alpha-parameter of the OU process, time scale of fluctuations.
            This parameter controls how long fluctuations are correlated in time ('dampening' parameter').
            Corresponds to 1/l in Rasmussen (2006), where l is the lengthscale.

        beta: float
            The beta-parameter of the OU process, frequency of the stochastic oscillation.
            This parameter regulates, with alpha, the peak-to-peak variability of oscillations.

        variance: float
            Variance of the OU process.
            This parameter controls the amplitude of the fluctuations.
            Corresponds to sigma^2 (not sigma!) in the Rasmussen (2006).

        noise: float
            The noise intrinsic to the collected observations y.
            If noise-free observations set noise = 0.0.

        oscillatory : boolean
            Default is False: constructs traces with a non-oscillatory Covariate_Matrix_OU
            True constructs trace with an oscillatory Covariance_Matrix_OU
        """
        self.alpha = alpha
        self.beta = beta
        self.variance = variance
        self.noise = noise
        self.oscillatory = oscillatory
        return

    def cov_matrix_ou(self, x1, x2, jitter = 0.0001):
        ''' Construct covariance matrix for the Ornstein-Uhlenbeck (OU) process.
            Defined by the Covariance Function of the OU process in Rasmussen (2006).

            This allows for both the oscillatory and non-oscillatory OU processes by choosing an appropriate beta.

        x1 : vector
            The vector x1 can be any set of data points

        x2: vector
            The vector x2 can be any set of data points, including x1 such that x2 = x1.

        Returns:
        --------

        Covariate_Matrix_OU: ndarray
            This array has the shape |x1|x|x2| where,
                the diagonal elements of Covariate_Matrix_OU are the variances of each data point;
                the non-diagonal elements of Covariate_Matrix_OU represent the correlation between data points.
        '''

        Nx1 = len(x1)
        Nx2 = len(x2)
        Covariate_Matrix_OU = np.zeros((Nx1, Nx2))
        for i in range(0, Nx1):
            for j in range(0, Nx2):
                if self.oscillatory:
                    oscillatory_term = np.cos(self.beta * abs(x1[i] - x2[j]))
                else:
                    oscillatory_term = 1
                Covariate_Matrix_OU[i, j] = self.variance * np.exp(-self.alpha * abs(x1[i] - x2[j]))*oscillatory_term
        return Covariate_Matrix_OU + np.eye(Nx1, Nx2)*jitter

    def generate_prior_ou_trace(self, duration, number_of_observations = 500, number_of_traces = 1):
        ''' Generate a sample from the Ornstein-Uhlenbeck (OU) process.
            Equivalent to taking a sample from the Prior f* ~ N(0, K(X*, X*))
        For reference, see for example the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

        Parameters:
        -----------

        duration : float
            time duration/length of the trace that should be generated.

        number_of_observations : integer
            how many time points should be sampled from the process

        number_of_traces : integer
            how many samples the function will take from the multivariate normal distribution and return as traces.

        Returns:
        --------

        trace : ndarray
            This array has shape (number_of_observations,number_of_traces + 1), where the first column contains time points, and the following columns
            contain function values of the OU process. The number of traces generated and hence the number of columns will depend on the specification given as input.
        '''

        timepoints = np.linspace(0.0,duration,number_of_observations)

        KXX = self.cov_matrix_ou(timepoints, timepoints)
        mean = np.zeros((len(timepoints)))

        path = np.random.multivariate_normal(mean, KXX, number_of_traces)

        trace = np.zeros((len(timepoints),number_of_traces+1))

        trace[:,0] = timepoints
        for index, path in enumerate(path):
            trace[:,index + 1] = path

        return trace

    def generate_predictor_and_posterior_ou_trace(self, observed_timepoints, observed_y, duration, number_of_observations = 500, number_of_traces = 1, cholesky_decompose = True, confidence_bounds = False):
            ''' Generate a posterior trace from the Ornstein-Uhlenbeck (OU) process.
                Equivalent to taking a sample from the Posterior GP distribution as follows

                f*|X*,X,y ~ N(K(X*,X) * [K(X,X) + I*noise**2]**-1 * y, K(X*, X*) - K(X*, X)[K(X,X)+I*noise**2]**-1 * K(X, X*))

                where K(.,.) is the Covariance Function evaluated for different inputs.
                      (This function yields appropriate covariance matrices for the OU oscillatory and non-oscillatory processes.)
                      X* are test time points
                      X are observed time points
                      y are observations

                Note: this expression accounts for noisy training data. To have noise-free predictions simply set noise = 0.0.
                Note: you can generate a trace for a predictor with no given observations.
                      To do so select [0.0] for both observed_timepoints and observed_y.

            For reference, see for example the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

            Parameters:
            -----------

            observed_timepoints: vector
                The vector of training inputs which have been observed.
                It takes vector inputs as a number of observations are needed to train the model well.
                Size of the vector |x| = N.
                observed_timepoints and observed_y must have the same size.

            observed_y: vector
                The vector of training inputs which have been observed. Can be noisy data.
                Size of the vector |y| = N.
                observed_timepoints and observed_y must have the same size.

            duration : float
                time duration/length of the trace that should be generated.

            number_of_observations : integer
                how many time points should be sampled from the process

            number_of_traces : integer
                how many samples the function will take from the multivariate normal distribution and return as sampled predictors.

            cholesky_decompose : boolean
                Default is True: constructs predictor and trains model using the cholesky decomposition of the Covariance Matrix.
                This is computationally more efficient and follows the Algorithm 2.1 as detailed in Rasmussen & Williams, 2006.

            Returns:
            --------

            trace : ndarray
                This array has shape (number_of_observations,number_of_traces+2), where the first column contains time points, the second column
                contains function values of the mean predictor of the OU process and the following column(s) contains function values of the sampled predictors.
            '''

            N = len(observed_timepoints)

            if len(observed_timepoints) == N and len(observed_y) == N:

                test_timepoints = np.linspace(0.0,duration,number_of_observations)
                posterior_trace = np.zeros((len(test_timepoints),number_of_traces+2))

                KXX = np.add(self.cov_matrix_ou(observed_timepoints, observed_timepoints), np.eye(len(observed_timepoints)) * self.noise**2)
                KXstarX = self.cov_matrix_ou(test_timepoints, observed_timepoints)
                KXXstar = self.cov_matrix_ou(observed_timepoints, test_timepoints)
                KXstarXstar = self.cov_matrix_ou(test_timepoints, test_timepoints)

                if cholesky_decompose:

                    L = cholesky(KXX, lower = True)

                    # Predictive Mean
                    alpha = cho_solve((L, True), observed_y)
                    mean_predictor = np.matmul(KXstarX, alpha)

                    # Predictive Variance
                    v = cho_solve((L, True), KXstarX.T)
                    cov_predictor = KXstarXstar - KXstarX.dot(v)

                    # Confidence Bounds
                    if confidence_bounds:
                        confidence_posterior = 1.96 * np.sqrt(np.diag(abs(cov_predictor)))

                    # Generate Trace and Store it
                    sampled_predictor = np.random.multivariate_normal(mean_predictor, cov_predictor, number_of_traces)

                    posterior_trace[:,0] = test_timepoints
                    posterior_trace[:,1] = mean_predictor
                    for index, path in enumerate(sampled_predictor):
                        posterior_trace[:,index+2] = path

                else: # direct analytic method

                    inverse = inv(KXX)
                    mean_star = np.zeros((len(test_timepoints))) ## Here taking mean = 0

                    mean_predictor = mean_star + np.matmul(np.matmul(KXstarX, inverse), observed_y)
                    cov_predictor = KXstarXstar - np.matmul(np.matmul(KXstarX, inverse), KXXstar)

                    if confidence_bounds:
                        confidence_posterior = 1.96 * np.sqrt(np.diag(abs(cov_predictor)))

                    sampled_predictor = np.random.multivariate_normal(mean_predictor, cov_predictor, number_of_traces)

                    posterior_trace[:,0] = test_timepoints
                    posterior_trace[:,1] = mean_predictor
                    for index, path in enumerate(sampled_predictor):
                        posterior_trace[:,index+2] = path

                if confidence_bounds:
                    return posterior_trace, confidence_posterior
                else:
                    return posterior_trace

            else:
                raise(ValueError("Observed Timepoints and Observed Y must be the same size."))

# todo : may make more sense to have a second class GP_SE
#        instead of containing the Squared Exponential functions in the detrending class
