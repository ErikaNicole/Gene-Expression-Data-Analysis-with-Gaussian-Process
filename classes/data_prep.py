# Class to prep cell and synthetic data for multiple purposes
import numpy as np
import pandas as pd
import os
from openpyxl import Workbook
from numpy.linalg import cholesky, det, lstsq, inv
from scipy.linalg import cholesky, cho_solve

    # todo: check possible small 'bug' in Detrending() __init__ inputs.
    #       I get a cholesky runtime error for the positive-definitiveness.
    #       I think it all comes down to the need for a 'jitter' in the evaluation of the covariance matrix
    #       In the Stan models a jitter was explicitly added and so it needs to be added here.
    #       Ask jochen. Shall I do so? Also there's reason to believe this needs to be checked elsewhere too.

class Data_Input():

    def __init__(self):
        return

    def get_excel(self):

        try:
            file_location = input(" Excel File Path : ")
            df = pd.read_excel(r'' + file_location)
        except:
            raise(FileNotFoundError("Ensure you specify full path, for example /Users/Me/Documents/file.xlsx"))

        return df

    def remove_nans(self, number_of_observed_cells, dataframe):

        observed_cells = []
        for cell in range(number_of_observed_cells):
            for observation in range(len(dataframe[2:])):                          # todo: change 2 to 1 when formatting convention has been finalised.
                if np.isfinite(np.array(dataframe.iloc[len(dataframe[2:]), cell + 1])):
                    print("Length of observations", len(dataframe[2:]), "for cell", cell + 1)
                    cells = dataframe.iloc[2:, cell + 1].to_numpy()
                    # plt.plot(cells)
                    observed_cells.append(np.array(cells))
                    break; # break is necessary to avoid second loop to run every time nan condition is met.
                elif np.isnan(np.array(dataframe.iloc[observation + 2, cell + 1])):
                    print("Length of observations", observation - 1, "for cell", cell + 1)  # Here -1 is to count the observation just before condition is met
                    cells = dataframe.iloc[2:(observation + 1), cell + 1].to_numpy()               # Here -1 as above as well as + 2 to count for the fact that we started from line 2.
                    observed_cells.append(np.array(cells))
                    break; # break is necessary to avoid second loop to run every time nan condition is met.

        return observed_cells

class Normalisation():

    def __init__(self):
        return

    def normalise_one_cell(self, cell_measurements):
        """
        Function to normalise cell data for plotting and evaluation purposes.
        :param cell_measurements: ndarray
        :return: normalised 1-D array of cell_measurements
        """
        try:
            y1 = np.zeros((len(cell_measurements)))
            for index, measurement in enumerate(cell_measurements):
                y1[index] = measurement - np.mean(cell_measurements)
            stds_y1 = np.std(y1)
            norm = y1/stds_y1
            return norm
        except Exception as e:
            print(str(e))

    def normalise_many_cells(self, cells_measurements):
        """
        To work this function requires to have an array cells_measurements correctly laid out.

        Parameters:
        -----------

        cells_measurements: ndarray
            nxm dnarray where n is the number of observations and m is the number is cells.
            Note that even if your cells have different # of measurements the code will work regardless.

        Returns:
        -----------

            norm: ndarray
                normalised nxm array of cells_measurements

        """

        number_of_cells = len(cells_measurements[0])
        number_of_measurements = len(cells_measurements[:,0])
        y = np.zeros((number_of_measurements,number_of_cells))
        stds_y = np.zeros((number_of_cells))

        for cell in range(len(cells_measurements[0])):                                  # Loops through columns ie different cells.
            for index, measurement in enumerate(cells_measurements[:,cell]):            # Loops through Individual Measurements per Cell.
                if measurement != 0:
                    y[index,cell] = measurement - np.mean(cells_measurements[:,cell])   # Stores Result of Normalization
            stds_y = np.std(y[:,cell])                                                  # Stores Standard Deviation per Cell

        norm = y/stds_y

        return norm

    def normalise_many_cells_from_list(self, cells_measurements):
        """
        To work this function requires to have an array cells_measurements correctly laid out.

        Parameters:
        -----------

        cells_measurements: list
            this list contains a number of arrays representative of the total number of observed cells.
            Each ndarray contained is a 1D vector of potentially different dimensions as the number of observations is not uniform.

        Returns:
        -----------

            norm: list
                normalised list of cells_measurements

        """

        number_of_cells = len(cells_measurements)
        normalised_cells = []
        #y = np.zeros((number_of_measurements,number_of_cells))
        #stds_y = np.zeros((number_of_cells))
        for cell in range(number_of_cells):
            number_of_measurements = len(cells_measurements[cell])
            normalised_cell = np.zeros(number_of_measurements)
            for index, measurement in enumerate(cells_measurements[cell]):
                normalised_cell[index] = measurement - np.mean(cells_measurements[cell])
            standard_deviation = np.std(cells_measurements[cell])
            normalised_cells.append(normalised_cell/standard_deviation)

        return normalised_cells

    # def noise(self, background_data):
    #    """
    #    :param normalised_cells_measurements:
    #    :return:
    #    """

    #    number_of_cells_background1 = len(background_data[0])
    #    number_of_measurements_background1 = len(background_data[:,0])
    #    y_background = np.zeros((number_of_measurements_background1,number_of_cells_background1))
    #    stds_background = np.zeros((number_of_cells_background1))
    #
    #    for cell in range(number_of_cells_background1):
    #        y_background[:,cell] = background_data[:,cell] - np.mean(background_data[:,cell])
    #        stds_background[cell] = np.std(y_background[:,cell])
    #
    #    # noise = np.mean(stds_background)/stds_y1
    #    # (mean of standard deviations of backgrounds)/(standard deviation of each cell)

    #    return noise

class Detrending():

    def __init__(self, alpha, variance = 1.0, noise = 0.001):
        """

        alpha: float
            The alpha-parameter of the SE process, time scale of fluctuations.
            This parameter controls how long fluctuations are correlated in time ('dampening' parameter').
            Corresponds to 1/l in Rasmussen (2006), where l is the lengthscale.

        variance: float
            Variance of the SE process.
            This parameter controls the amplitude of the fluctuations.
            Corresponds to sigma^2 (not sigma!) in the Rasmussen (2006).

        """
        self.alpha = alpha
        self.variance = variance
        self.noise = noise
        return

    def cov_matrix_SE(self, x1, x2):
        ''' Construct Covariance Matrix using the Squared Exponential (SE) kernel.
            For reference, see the book Rasmussen (2006) 'Gaussian Processes for Machine learning'

        x1 : vector
            The vector x1 can be any set of data points

        x2: vector
            The vector x2 can be any set of data points, including x1 such that x2 = x1.

        Returns:
        --------

        Covariate_Matrix_SE: ndarray
            This array has the shape |x1|x|x2| where,
                the diagonal elements of Covariate_Matrix_SE are the variances of each data point;
                the non-diagonal elements of Covariate_Matrix_SE represent the correlation between data points.
        '''

        Nx1 = len(x1)
        Nx2 = len(x2)
        Covariate_Matrix_SE = np.zeros((Nx1, Nx2))
        for i in range(0, Nx1):
            for j in range(0, Nx2):
                Covariate_Matrix_SE[i, j] = self.variance * np.exp(- self.alpha * (x1[i] - x2[j])**2)

        return Covariate_Matrix_SE

    def generate_prior_SE_trace(self, test_timepoints, number_of_traces = 1):
        ''' Generate a sample from the Squared Exponential (SE) Covariate Matrix.
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
            This array has shape (number_of_observations,2), where the first column contains time points, and the second columns
            contains function values of the OU process.
        '''

        KXX = self.cov_matrix_SE(test_timepoints, test_timepoints)
        mean = np.zeros((len(test_timepoints)))

        path = np.random.multivariate_normal(mean, KXX, number_of_traces)

        trace = np.zeros((len(test_timepoints),number_of_traces+1))

        trace[:,0] = test_timepoints
        for index, path in enumerate(path):
            trace[:,index + 1] = path

        return trace

    def fit_SE(self, observed_timepoints, observed_y, test_timepoints, number_of_traces = 1, cholesky_decompose = True): #,duration = None, number_of_observations = None):
            ''' Generate a posterior trace from the Squared Exponential (Se) Covariate Function.
                Equivalent to taking a sample from the Posterior GP distribution as follows

                f*|X*,X,y ~ N(K(X*,X) * [K(X,X) + I*noise**2]**-1 * y, K(X*, X*) - K(X*, X)[K(X,X)+I*noise**2]**-1 * K(X, X*))

                where K(.,.) is the SE Covariance Function evaluated for different inputs.
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

                posterior_trace = np.zeros((len(test_timepoints),number_of_traces+2))

                KXX = np.add(self.cov_matrix_SE(observed_timepoints, observed_timepoints), np.eye(len(observed_timepoints)) * self.noise**2)
                KXstarX = self.cov_matrix_SE(test_timepoints, observed_timepoints)
                KXXstar = self.cov_matrix_SE(observed_timepoints, test_timepoints)
                KXstarXstar = self.cov_matrix_SE(test_timepoints, test_timepoints)

                if cholesky_decompose:

                    L = cholesky(KXX, lower = True)

                    # Predictive Mean
                    alpha = cho_solve((L, True), observed_y)
                    mean_predictor = np.matmul(KXstarX, alpha)
                    mean_predictor = mean_predictor.ravel() # Necessary to flatted 2d in 1d

                    # Predictive Variance
                    v = cho_solve((L, True), KXstarX.T)
                    cov_predictor = KXstarXstar - KXstarX.dot(v)

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

                    sampled_predictor = np.random.multivariate_normal(mean_predictor, cov_predictor, number_of_traces)

                    posterior_trace[:,0] = test_timepoints
                    posterior_trace[:,1] = mean_predictor
                    for index, path in enumerate(sampled_predictor):
                          posterior_trace[:,index+2] = path

                return posterior_trace

            else:
                raise(ValueError("Observed Timepoints and Observed Y must be the same size."))

    def detrend_data(self, observed_timepoints, data_to_detrend, cholesky_decompose = True):
        """
        To work this function requires to have an array data_to_detrend correctly laid out.

        Parameters:
        -----------

        observed_timepoints: vector
            The vector of training inputs which have been observed.
            It takes vector inputs as a number of observations are needed to train the model well.
            Size of the vector |x| = N.

        data_to_detrend: ndarray
            nxm ndarray where the |m| is the number of cells to detrend and |n| is the number of observations.

        Returns:
        -----------

        detrended_data: ndarray
            nxm ndarray where the |m| is the number of cells to detrend and |n| is the number of observations.
            The entries have all been detrended using the respective fitted GP using the SE Covariate Function.

        """

        duration = round(float(observed_timepoints[-1]))

        test_timepoints = np.linspace(0, duration, 500)
        test_timepoints = np.unique(np.sort(np.hstack((test_timepoints, observed_timepoints.ravel()))))
        # This way we make sure test_timepoints contain a number of timepoints, including the observed ones!
        # But also removing duplicates just in case.

        # Note: try except wrap is needed to deal with the detrending of one cell.

        try:
            number_of_cells = len(data_to_detrend[0])
            number_of_observations = len(data_to_detrend[:,0])
            fits = np.zeros((len(test_timepoints), number_of_cells))

            # First loop is made to carry out different fittings for each cell, hence yielding different predictors.
            for cell in range(number_of_cells):
                observed_cell = data_to_detrend[:,cell]
                # I want to ideally have the fit to be evaluated at more points than my actual observations.
                # I can then detrend and hopefully get my timepoints to match up.
                fits[:, cell] = self.fit_SE(observed_timepoints, observed_cell, test_timepoints, 1, cholesky_decompose)[:,1]

            # Of course predictor has a lot more points than the observed one,
            # This loop gives me the indices of test_timepoints corresponding to the equivalent time point observed.
            # Using the corresponding indices we can then carry out the detrending.

            detrended_data = np.zeros((number_of_observations, number_of_cells))
            for i, timepoint1 in enumerate(test_timepoints):
                for j, timepoint2 in enumerate(observed_timepoints):
                    if timepoint1 == timepoint2:
                        # match = [j, i]
                        # matched_index = np.append(matched_index, match, axis = 0)
                        detrended_data[j, :] = data_to_detrend[j, :] - fits[i, :]

        except:

            number_of_cells = 1
            number_of_observations = len(data_to_detrend)
            fits = self.fit_SE(observed_timepoints, data_to_detrend, test_timepoints, 1, cholesky_decompose)[:,1]
            detrended_data = np.zeros((number_of_observations, number_of_cells))
            for i, timepoint1 in enumerate(test_timepoints):
                for j, timepoint2 in enumerate(observed_timepoints):
                    if timepoint1 == timepoint2:
                        if data_to_detrend[j] != 0:
                            # match = [j, i]
                            # matched_index = np.append(matched_index, match, axis = 0)
                            detrended_data[j] = data_to_detrend[j] - fits[i]

        return detrended_data, test_timepoints, fits

    def detrend_data_from_list(self, observed_timepoints, data_to_detrend, cholesky_decompose = True):
        """
        To work this function requires to have an array data_to_detrend correctly laid out.

        Parameters:
        -----------

        observed_timepoints: vector
            The vector of training inputs which have been observed.
            It takes vector inputs as a number of observations are needed to train the model well.
            Size of the vector |x| = N.

        data_to_detrend: list
            This list contains a number of arrays representative of the total number of observed cells.
            Each ndarray contained is a 1D vector of potentially different dimensions as the number of observations is not uniform.


        Returns:
        -----------

        detrended_data: list
            This list contains a number of arrays representative of the total number of observed cells.
            The entries have all been detrended using the respective fitted GP using the SE Covariate Function.

        """

        duration = round(float(observed_timepoints[-1]))

        test_timepoints = np.linspace(0, duration, 500)
        test_timepoints = np.unique(np.sort(np.hstack((test_timepoints, observed_timepoints.ravel()))))
        # This way we make sure test_timepoints contain a number of timepoints, including the observed ones!
        # But also removing duplicates just in case.
        number_of_cells = len(data_to_detrend)

        # Note: try except wrap is needed to deal with the detrending of one cell.

        if number_of_cells != 1:

            fits = []

            # First loop is made to carry out different fittings for each cell, hence yielding different predictors.
            for cell in range(number_of_cells):
                number_of_observations = len(data_to_detrend[cell]) # Each cell has a different number of observations.
                observed_cell = data_to_detrend[cell]
                observed_timepoints_per_cell = observed_timepoints[0:number_of_observations]
                # I want to ideally have the fit to be evaluated at more points than my actual observations.
                # I can then detrend and hopefully get my timepoints to match up.
                fits.append(self.fit_SE(observed_timepoints_per_cell, observed_cell, test_timepoints, 1, cholesky_decompose)[:,1])

            # Of course predictor has a lot more points than the observed one,
            # This loop gives me the indices of test_timepoints corresponding to the equivalent time point observed.
            # Using the corresponding indices we can then carry out the detrending.

            detrended_data = []
            #detrended_data = np.zeros((number_of_observations, number_of_cells))
            for cell in range(number_of_cells):
                number_of_observations = len(data_to_detrend[cell])
                detrended = np.zeros(number_of_observations)
                observed_timepoints_per_cell = observed_timepoints[0:number_of_observations]
                for i, timepoint1 in enumerate(test_timepoints):
                    for j, timepoint2 in enumerate(observed_timepoints_per_cell):
                        if timepoint1 == timepoint2:
                            detrended[j] = data_to_detrend[cell][j] - fits[cell][i]
                detrended_data.append(detrended)

        else:

            observed_cell = data_to_detrend[0]
            number_of_observations = len(observed_cell)
            observed_timepoints = observed_timepoints[0:number_of_observations]
            fits = self.fit_SE(observed_timepoints, observed_cell, test_timepoints, 1, cholesky_decompose)[:,1]
            detrended = np.zeros(number_of_observations)
            detrended_data = []
            for i, timepoint1 in enumerate(test_timepoints):
                for j, timepoint2 in enumerate(observed_timepoints):
                    if timepoint1 == timepoint2:
                        detrended[j] = observed_cell[j] - fits[i]
            detrended_data.append(detrended)

        return detrended_data, test_timepoints, fits

class Data_Export():

    def __init__(self):
        return

    def list_of_cells(self, number_of_observed_cells):

        list = []
        array = np.arange(0, number_of_observed_cells, 1)
        s = "Cell"
        for item in array:
            s += str(item + 1)
            list.append(s)
            s = "Cell"

        return list

    def export_detrended_data(self, detrended_data):
        """
        Function to export into dataframe

        Parameters
        ---------

        detrended_data: ndarray
            Use directly output from detrend_data (ndarray output) or detrend_data_from_list (list output) functions

        Returns
        --------
        dataframe: DataFrame

        """

        if isinstance(detrended_data, np.ndarray):
            number_of_observed_cells = len(detrended_data[0, :])
            column_names = self.list_of_cells(number_of_observed_cells)
            dataframe = pd.DataFrame(data = detrended_data, columns = column_names)
        else:
            number_of_observed_cells = len(detrended_data)
            lengths = np.zeros((number_of_observed_cells))
            column_names = self.list_of_cells(number_of_observed_cells)
            for cell in range(number_of_observed_cells):
                lengths[cell] = len(detrended_data[cell])
            length = max(lengths)
            export_detrended_data = np.zeros((int(length), number_of_observed_cells))
            for cell in range(number_of_observed_cells):
                export_detrended_data[:, cell] = np.hstack((detrended_data[cell], np.repeat("nan", length - len(detrended_data[cell]))))

            dataframe = pd.DataFrame(data = export_detrended_data, columns = column_names)

        return dataframe

    def export_parameter_estimates(self, parameters):
        """
        Function to export into dataframe

        parameters:
            Use directly output [1] from model_selection

        Returns
        ------
        dataframe: DataFrame

        """

        number_of_observed_cells = len(parameters[0, :])
        column_names = self.list_of_cells(number_of_observed_cells)
        row_names = ["OUosc alpha", "OUosc beta", "OUosc variance", "OUosc noise", "OU alpha", "OU beta", "OU variance", "OU noise"]
        dataframe = pd.DataFrame(data = parameters, index = row_names, columns = column_names)
        dataframe = dataframe.drop(index = ['OU beta'])

        return dataframe

    def export_LLRs_estimates(self, LLRs):
        """
        Function to export into dataframe

        LLRs:
            Use directly output [1] from model_selection

        Returns
        -------
        dataframe: DataFrame

        """

        number_of_total_cells = len(LLRs[1])
        length = max(len(LLRs[0]), len(LLRs[1]))
        LLRs[0] = np.hstack((LLRs[0], np.repeat("nan", length - len(LLRs[0]))))
        LLRs = np.vstack((LLRs[0], LLRs[1]))
        column_names = self.list_of_cells(number_of_total_cells)
        row_names = ["LLR Observed", "LLR Synthetic"]
        dataframe = pd.DataFrame(data = LLRs, index = row_names, columns = column_names)

        return dataframe

    def export_q_values(self, q_values):
        """
        Function to export into dataframe

        Parameters
        -------

        q-values: ndarray
            Use directly output [1] from model_selection

        Returns
        -------
        dataframe: DataFrame

        """

        number_of_total_cells = len(q_values[0])
        column_names = self.list_of_cells(number_of_total_cells)
        row_names = ["q_values", "pass list"]
        dataframe = pd.DataFrame(data = q_values, index = row_names, columns = column_names)

        return dataframe

    def period_estimation(self, parameters):
        """
        Function to export into dataframe

        parameters:
            Use directly output [1] from model_selection

        Returns
        ------
        dataframe: DataFrame

        """

        number_of_total_cells = len(parameters[0])
        estimated_betas = parameters[1,:]
        estimated_periods = (1/estimated_betas)
        column_names = self.list_of_cells(number_of_total_cells)
        data = np.vstack((estimated_betas, estimated_periods))
        row_names = ["Estimated Frequency (1/hours)", "Estimated Period (hours)"]
        dataframe = pd.DataFrame(data = data, index = row_names, columns = column_names)

        return dataframe

    def export_contingency_table(self, contigency_table):
        """

        :param contigency_table:
        :return:
        """

        column_names = ["Condition Positive", "Condition Negative"]
        row_names = ["Predicted Condition Positive", "Predicted Condition Negative"]
        dataframe = pd.DataFrame(data = contigency_table, index = row_names, columns = column_names)

        return dataframe





##### TESTING DETREND_DATA

### Data Set Up

#X = np.linspace(0, 100, 500)
#Y_m = np.sin(0.5*X)
#Y = Y_m + np.random.normal(0, 0.5, len(Y_m)) # + noise
#observed_id = np.arange(0, 500, 10)
#observed_cell_x = X[observed_id]
#observed_cell_y = Y[observed_id]

#observed_cell_y = np.array([observed_cell_y]).T
#observed_cell_x = np.array([observed_cell_x]).T
#data = np.hstack((observed_cell_y, observed_cell_y))
#data = np.hstack((data, observed_cell_y))
## Make trended data
#for i in range(50):
    #data[i,0] = data[i,0] + i*1.1

### Calling Function INPUTS

#test_timepoints= np.linspace(0, 100, 500)

#detrend = Detrending(0.0005, 0.5, 0.0001)
#visualisation = Visualisation_GP(0.0005, 0.5, 0.5, 0.0001, True, observed_cell_x, observed_cell_y, True)
#visualisation.gp_se_trace_plot(test_timepoints, number_of_traces = 1)
#detrended_data = detrend.detrend_data(observed_cell_x, observed_cell_y)







#for cell in range(len(observed_cell_y[0])):
    #observations = observed_cell_y[:,cell]
    #detrend.fit_SE(observed_cell_x, observations, 100)

#for cell in range(len(data[0])):
    #observations = data[:,cell]
    #print(detrend.fit_SE(observed_cell_x, observations, 100))
