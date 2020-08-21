# Class to carry out Hyper Parameter Estimation

import classes.gp as gp
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import cholesky, det, lstsq, inv
from scipy.linalg import cholesky, cho_solve
from functools import partial

optimizereval = 1 # re-run every time before running the optimizer

class Optimisation():

    def __init__(self, oscillatory, observed_timepoints, observed_y):
        """
        oscillatory: boolean
            Default is False: computes non-oscillatory Covariate_Matrix_OU.
            True constructs oscillatory Covariance_Matrix_OU.

        observed_timepoints : vector
            The vector of training inputs which have been observed.
            It takes vector inputs as a number of observations are needed to train the model well.
            Size of the vector |x| = N.
            observed_x and observed_y must have the same size.

        observed_y: vector
            The vector of training inputs which have been observed. Can be noisy data.
            Size of the vector |y| = N.
            x and y must have the same size.
        """
        self.oscillatory = oscillatory
        self.observed_timepoints = observed_timepoints
        self.observed_y = observed_y
        return

    def neg_marginal_loglikelihood(self, theta, cholesky_decompose = True):
        '''
        Returns a function that computes the negative log marginal likelihood for training data observed_x and observed_y.

        Parameters:

            theta: ndarray
            The vector containing the values to be updated for the optimization algorithm chosen to find an optima.
            Such values, being the hyper parameters, are always printed in this order:
                                [alpha, beta, variance, noise]

            cholesky_decompose : boolean
            Default is True: constructs predictor and trains model using the cholesky decomposition of the Covariance Matrix.
            This is computationally more efficient and follows the Algorithm 2.1 as detailed in Rasmussen & Williams, 2006.

        Returns:
            Minimization objective.
        '''

        N = len(self.observed_timepoints)

        if len(self.observed_timepoints) == N and len(self.observed_y) == N:

            self.alpha, self.beta, self.variance, self.noise = theta[0], theta[1], theta[2], theta[3]
            self.GP = gp.GP(self.alpha, self.beta, self.variance, self.noise, self.oscillatory)

            cov_matrix_y = np.add(self.GP.cov_matrix_ou(self.observed_timepoints, self.observed_timepoints), np.eye(len(self.observed_timepoints)) * self.noise**2)

            if cholesky_decompose:

                L = cholesky(cov_matrix_y, lower = True)
                alpha = cho_solve((L, True), self.observed_y)
                return -(-0.5*self.observed_y.T.dot(alpha) - np.trace(np.log(L)) - N/2.0*np.log(2*np.pi))

            else:

                return 0.5 * np.log(det(cov_matrix_y)) + 0.5 * self.observed_y.T.dot(inv(cov_matrix_y).dot(self.observed_y)) + 0.5 * len(self.observed_timepoints) * np.log(2*np.pi)

        else:

            raise(ValueError("Observed Timepoints and Observed Y must be the same size."))

    def optimizing_neg_marginal_loglikelihood(self, start_values, method = 'L-BFGS-B', bounds = ((1e-10, None), (None, None), (1e-10, None), (1e-10, None)), cholesky_decompose = True):
        ''' Minimises the negative marginal log likelihood with respect to the Hyper Parameters of the OU Function Matrix.
            The parameters estimated are alpha, beta and variance respectively which are returned by the optimizer.

                        Hyper parameters in both inputs and outputs are always printed in this order:
                                                [alpha, beta, variance, noise]

            The optimization occurs via the scipy.optimize.minimize library.
            For reference check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            The optimization takes place on the function 'neg_marginal_loglikelihood'.

        start_values: vector
            The starting points for the optimization algorithm chosen.
            Due to risk of optimizer incurring in local minima it is important a sensible choice of starting values is made.
            Note that the start_values vector should be of size 3, one per hyper parameter.

            Note: if not oscillatory beta will not affect inference however the start_values entry should still be included:
            start_values = [0.5, 0.0, 0.5, 0.0]

        cholesky_decompose : boolean
            Default is True: constructs predictor and trains model using the cholesky decomposition of the Covariance Matrix.
            This is computationally more efficient and follows the Algorithm 2.1 as detailed in Rasmussen & Williams, 2006.

            todo: doublecheck with Jochen about this point. Keeps coming up.
            Warning: if incurring in error numpy.linalg.LinAlgError: -th leading minor of the array is not positive definite
                     you may take that as being an error with reference to the covariance matrix and the function struggling
                     to get a cholesky decomposition of it, as the OU Process produces a semi-positive definite matrix.

        method: string
            method used in optimizer to carry out computation.
            For a complete list of methods supported by optimizers consult:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            Default is L-BFGS-B.

        bounds: tuple
            Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.

            How to specify the bounds:
            n-tuple must have n to be equivalent to the number of parameters estimated.
            Hence, here tuple must be of size 4 for alpha, beta, variance and noise.
            Construct as a sequence of (min, max) pairs for each element to be estimated.

            Default is None, which is used to specify no bound.

            Note:
                Bounds given for L-BFGS-B Method such that parameters are strictly positive - always yields successful result.
                Bounds given for TNC Method such that parameters are strictly positive - always yields successful result.

                Example Bounds - ((1e-10, None), (1e-10, None), (1e-10, None), (1e-10, None))

                Generally - Setting strictly positive (1e-10, None) for alpha stops the optimizer from failing in most cases.

        Returns:
        --------

        res : OptimizeResult Object
            Returns the object of an optimization run, for more details on each outputs meaning check out:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            Of particular importance:
            fun: specifies the optimized negative log marginal likelihood for the training data specified and the chosen
                 hyper parameters.
            x:   estimated hyper parameters, alpha, beta and the variance respectively.

            To extract: hyperparameters estimates from OptimizeResult Object use .x
        '''

        if len(start_values) != 4:
            raise(ValueError('start_values input must be of size 4 for alpha, beta, variance and noise respectively.'))

        if isinstance(method, str):
            if method == 'L-BFGS-B' or method == 'TNC' or method == 'Powell' or method == 'SLSQP':
                res = minimize(self.neg_marginal_loglikelihood, args = (cholesky_decompose), x0 = start_values, bounds = bounds, method = method)
            else:
                if bounds != None:
                    raise(ValueError('Bounds cannot be passed with this method.'))
                res = minimize(self.neg_marginal_loglikelihood, args = (cholesky_decompose), x0 = start_values, method = method)
        else:
            raise(ValueError('method input must be a string.'))
        return res

    # Hyper Parameter Estimation (With Callback & History)
    def optimizing_neg_marginal_loglikelihood_callback(self, start_values, method = 'L-BFGS-B', bounds = ((1e-10, None), (None, None), (1e-10, None), (None, None)), cholesky_decompose = True, callback = True):
        ''' Minimises the negative marginal log likelihood with respect to the Hyper Parameters of the OU Function Matrix.
            The parameters estimated are alpha, beta and variance respectively which are returned by the optimizer.

                        Hyper parameters in both inputs and outputs are always printed in this order:
                                                [alpha, beta, variance, noise]

            The optimization occurs via the scipy.optimize.minimize library.
            For reference check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            Note: The optimization takes place on the partial function 'partial_neg_marginal_loglikelihood'.
                  The reason for that is purely to allow the callback function to run smoothly.
                  The result is identical either way.

        start_values: vector
            The starting points for the optimization algorithm chosen.
            Due to risk of optimizer incurring in local minima it is important a sensible choice of starting values is made.
            Note that the start_values vector should be of size 4, one per hyper parameter.

            Note: if not oscillatory beta will not affect inference however the start_values entry should still be included:
            start_values = [0.5, 0.0, 0.5, 0.0]

        cholesky_decompose : boolean
            Default is True: constructs predictor and trains model using the cholesky decomposition of the Covariance Matrix.
            This is computationally more efficient and follows the Algorithm 2.1 as detailed in Rasmussen & Williams, 2006.

        method: string
            method used in optimizer to carry out computation.
            For a complete list of methods supported by optimizers consult:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            Default is L-BFGS-B.

        bounds: tuple
            Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.

            How to specify the bounds:
            n-tuple must have n to be equivalent to the number of parameters estimated.
            Hence, here tuple must be of size 4 for alpha, beta, variance and noise.
            Construct as a sequence of (min, max) pairs for each element to be estimated.

            Default is None, which is used to specify no bound.

            Note:
                Bounds given for L-BFGS-B Method such that parameters are strictly positive - always yields successful result.
                Bounds given for TNC Method such that parameters are strictly positive - always yields successful result.

                Example Bounds - ((1e-10, None), (1e-10, None), (1e-10, None), (1e-10, None))

                Generally - Setting strictly positive (1e-10, None) for alpha stops the optimizer from failing in most cases.

            Warning: if incurring in error numpy.linalg.LinAlgError: -th leading minor of the array is not positive definite
                     you may take that as being an error with reference to the covariance matrix and the function struggling
                     to get a cholesky decomposition of it, as the OU Process produces a semi-positive definite matrix. To
                     steer away from it we have to set some bounds values, which seem to take care of the problem.

        Returns:
        --------
        todo: check the convention for when returning functions

        res : OptimizeResult Object
            Returns the object of an optimization run, for more details on each outputs meaning check out:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            Of particular importance:
            fun: specifies the optimized negative log marginal likelihood for the training data specified and the chosen
                 hyper parameters.
            x:   estimated hyper parameters, alpha, beta and the variance respectively.

            To extract: hyperparameters estimates from OptimizeResult Object use .x
        '''

        partial_neg_marginal_likelihood = partial(self.neg_marginal_loglikelihood, cholesky_decompose = cholesky_decompose)

        if callback:
            history = []
            print("Callback for the Optimisation Function:")
            #print("Iter", " -", "Alpha", "   - ", "Beta", "  -   ", "Variance", " - ", "Noise", "  -  ", "Log Marginal Likelihood")
            print('{:7} {:11} {:11} {:11} {:12} {}'.format("Iter", "Alpha", "Beta", "Variance", "Noise", "Log Marginal Likelihood"))
            def callbackfunction(theta):
                """
                Generate a callback that prints every iteration

                    iteration number | parameter values | objective function

                and stores all the parameter values in history
                """
                global optimizereval
                thetas = [theta[0], theta[1], theta[2], theta[3], partial_neg_marginal_likelihood(theta)]
                history.append(thetas)
                print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}    {5: 3.6f}'.format(optimizereval, theta[0], theta[1], theta[2], theta[3], partial_neg_marginal_likelihood(theta)))
                optimizereval += 1
        else:
            callbackfunction = None

        if len(start_values) != 4:
            raise(ValueError('start_values input must be of size 4 for alpha, beta, variance and noise respectively.'))

        if isinstance(method, str):
            if method == 'L-BFGS-B' or method == 'TNC' or method == 'Powell' or method == 'SLSQP':
                res = minimize(partial_neg_marginal_likelihood, x0 = start_values, bounds = bounds, method = method, callback = callbackfunction)
            else:
                if bounds != None:
                    raise(ValueError('Bounds cannot be passed with this method.'))
                res = minimize(partial_neg_marginal_likelihood, x0 = start_values, method = method, callback = callbackfunction)
        else:
            raise(ValueError('method input must be a string.'))

        if callback:
            return res, np.array(history)
        else:
            return res




