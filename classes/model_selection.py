import matplotlib.pyplot as plt
import numpy as np
import time
import csaps
import classes.data_prep as prep
import classes.gp as gp
import classes.data_visualisation as visualisation
import classes.optimisation as optimisation
import classes.gillespie as gillespie

# todo: Currently holding a Model Selection thought: shall I have control cells as inputs too such that they can help balance
#       the model selection process? Or, if I run two separate model selections for control vs observed data does that change inference?

class ModelSelection():

    def __init__(self):
        self.visualisation = visualisation.Visualisation_ModelSelection()
        return

    def approximation_of_LLRs(self, observed_timepoints, cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess): #, normalise = True):
        '''
        Choosing a threshold for the LLR relies on a Bootstrap approach to approximate the LLR distribution
        for a population of non-oscillating cells:

        1. Fit OUosc and OU GP, Storing Results and Calculating LLR from each Observed Cell
        2. Simulate Synthetic Cells with the Null Aperiodic OU Model.
        3. Calculate the LLRs for each Synthetic Cell and store them.
            The number of synthetic cells generated is equally
        4. Normalise the LLRs.
        5. Optional: plot a histogram of the LLRs).
            this gives an approximate distribution for the LLRs which allow us to calculate

        Corresponds to step 1. and 2. in original paper.

        Parameters:
        -----------

        observed_timepoints: ndarray
            Input a nx1 array where n is the number of observations per cell. This will represent the timepoints at which
            the gene expression levels for each cell were measured.

        cells : ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file or from the output of GP.generate_prior_ou_trace().

        number_of_observed_cells: integer
            Might not need this as can be retrieved from the cells array but could also leave it as a failsafe that it
            coincides with the number of columns.

        number_of_synth_cells: integer
            As a general rule of thumb, as proposed by 'Identifying stochastic oscillations in single-cell live imaging time
            series using Gaussian processes', it is preferred to pick out at least 20 synthetic cells per observed cell.
            Hence if the number of observed cells is 10, we get a total of 200 synthetic cells to calculate LLRs from.

        normalised: boolean
            In project LLRs from data are normalised before moving forward. Hence default is True.
            Note that in the project only the observed one seem to have been normalised, not the synthetic bootstrap.

        Returns:
        --------

        LLR: ndarray
        Returns an array of size (2,) where
            LLR[0]: represents the normalised LLR values of the time series from the data.
            LLR[1]: represents the LLR values of the synthetic time series from the bootstrapping.
        '''

        start_time = time.time()

        number_of_parameters = 4  # As we have alpha, beta and variance and noise
        number_of_observations = len(observed_timepoints)

        par_cell_non_osc = np.zeros((number_of_parameters, number_of_observed_cells))
        loglik_cell_non_osc = np.zeros((number_of_observed_cells))
        loglik_cell_osc = np.zeros((number_of_observed_cells))
        llikelihood_ratio_observed_cells = np.zeros((number_of_observed_cells))
        normalised_llikelihood_ratio_observed_cells = np.zeros((number_of_observed_cells))

        # First Loop Deals Fitting OUosc and OU GP, Storing Results and Calculating LLR from each Observed Cell

        for i in range(number_of_observed_cells):

            self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints, observed_y = cells[:,i])
            op_cell_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess)
            self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints, observed_y = cells[:,i])
            op_cell_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess)

            par_cell_non_osc[:,i] = op_cell_non_osc.x
            loglik_cell_non_osc[i] = - op_cell_non_osc.fun
            loglik_cell_osc[i] = - op_cell_osc.fun
            llikelihood_ratio_observed_cells[i] = 2*(loglik_cell_osc[i] - loglik_cell_non_osc[i])
            # Normalisation as described in project is (LLR/length of data) * 100
            normalised_llikelihood_ratio_observed_cells[i] = np.divide(llikelihood_ratio_observed_cells[i],number_of_observed_cells*number_of_observations)*100

        print("Completed Fitting of", number_of_observed_cells, "Observed Cells")

        # Second Loop Deals with Synthetic Cells Generation to Approximate LLR Distribution

        number_of_synth_cell_per_observed_cell = round(number_of_synthetic_cells/number_of_observed_cells)
        # synthetic_data = np.zeros((number_of_observations, number_of_synth_cell_per_observed_cell)) should not be needed
        synthetic_cells = np.zeros((number_of_observed_cells, number_of_observations, number_of_synth_cell_per_observed_cell))
        for cell in range(number_of_observed_cells):
            self.gp = gp.GP(alpha = par_cell_non_osc[0,cell], beta = par_cell_non_osc[1,cell], variance = par_cell_non_osc[2,cell], noise = par_cell_non_osc[3, cell], oscillatory = False)
            synthetic_data = self.gp.generate_prior_ou_trace(duration = observed_timepoints[-1], number_of_observations = number_of_observations, number_of_traces = number_of_synth_cell_per_observed_cell)[:,1:number_of_synth_cell_per_observed_cell+1]
            synthetic_cells[cell, :, :] = np.array([synthetic_data])
            # print("Completed Generation of Synthetic Cells of Observed Cell", cell + 1)

        print("Completed Generation of Synthetic Cells for all", number_of_observed_cells," Observed Cell")

        # Third Loop Deals With Calculating LLR for each new Synthetic Cell

        llikelihood_ratio_synth_cells = np.zeros((number_of_synth_cell_per_observed_cell*number_of_observed_cells))
        i = -1
        for cell in range(number_of_observed_cells):
            for synth_cell in range(number_of_synth_cell_per_observed_cell):
                i = i + 1
                self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints, observed_y = synthetic_cells[cell,:,synth_cell])
                neg_llik_synth_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess).fun
                self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints, observed_y = synthetic_cells[cell,:,synth_cell])
                neg_llik_synth_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess).fun
                llikelihood_ratio_synth_cells[i] = 2*((-neg_llik_synth_osc) - (-neg_llik_synth_non_osc))
            # print("Completed Fitting of OUosc and OU GP on Synthetic, Storing Results and Calculating LLR from Observed Cell", cell + 1)

        print("Completed Fitting of OUosc and OU GP on", number_of_synthetic_cells," Synthetic Cells, storing their results and calculating LLRs")

        # If Statement Deals with Normalising LLRs

        # Don't think this is needed anymore, normalisation included in first for loop
        # if normalise:
        #    llikelihood_ratio_observed_cells_normalised = np.divide(llikelihood_ratio_observed_cells,number_of_observed_cells*number_of_observations)*100
        #    #llikelihood_ratio_synthetic_cells_normalised = np.divide(llikelihood_ratio_synth_cells,number_of_synthetic_cells*number_of_observations)*100
        #    LLR = np.array((llikelihood_ratio_observed_cells_normalised, llikelihood_ratio_synth_cells)) #, llikelihood_ratio_synthetic_cells_normalised))
        #else:
        #    LLR = np.array((llikelihood_ratio_observed_cells, llikelihood_ratio_synth_cells))

        LLR = np.array((normalised_llikelihood_ratio_observed_cells, llikelihood_ratio_synth_cells))

        end_time = time.time()
        print("Execution time:", end_time-start_time)

        return LLR

    def distribution_of_proportion_of_non_osc_cells(self, LLR, number_of_observed_cells, number_of_synthetic_cells):
        '''
        Estimating Proportion pi_0 of non oscillating cells by comparing the shape of the data set with that of
        non oscillating cells generated with the bootstrap.

        To find pi_0 the following must be calculated:

                       pi_0 = (# LLR_data < lambda)/number_of_observed_cells / (# LLR_synth < lambda)/number_of_synthetic_cells

        Here lambda is a tuning parameter, contained in a vector tuning_parameter.
        This is kept between a range going from the minimum LLR value to the maximum observed LLR value.
        The above formulae is calculated once for each tuning parameter.
        This gives a range of possible pi0 which will allows us to pick out an estimate from the distribution of pi_0 observed.

        Corresponds to step 3. in original paper (without spline fitting).

        Parameters:
        -----------

        LLR: ndarray
            Use output directly from the previous function approximation_of_LLRs

        number_of_observed_cells: integer
            As specified when called the previous function approximation_of_LLRs

        number_of_synthetic_cells: integer
            As specified when called the previous function approximation_of_LLRs

        Returns:
        --------

        pi_0: ndarray
            Observed distribution of proportion of non-oscillating cells for different tuning parameter values.
        '''

        #Extract LLRs from the vector.

        LLR_data = np.sort(LLR[0])
        LLR_synth = np.sort(LLR[1])

        # Paper Suggests Tuning Parameter should be a range between minimum and maximum LLRs.
        # Note not specified whether max and min of population of non-oscillating cells or all cells including observed ones.
        # But I tried with the synth data only and its definitely better including both.

        max_LLR = max(max(LLR_data), max(LLR_synth))
        min_LLR = min(min(LLR_data), min(LLR_synth))
        #In original code included the following line? A little confused as to why its necessary
        min_LLR = max_LLR - 0.9*(max_LLR - min_LLR)

        tuning_parameter = np.linspace(min_LLR, max_LLR, 50)
        size = len(tuning_parameter)
        count_data = np.zeros((size))
        count_synth = np.zeros((size))
        pi_0 = np.zeros((size))

        for index, tuning_parameter in enumerate(tuning_parameter):
            count_data[index] = sum(1 for LLR in LLR_data if LLR < tuning_parameter)
            count_synth[index] = sum(1 for LLR in LLR_synth if LLR < tuning_parameter)
            pi_0[index] = (count_data[index]/number_of_observed_cells)/(count_synth[index]/number_of_synthetic_cells)

        return pi_0

    def estimation_of_proportion_of_non_osc_cells(self, LLRs, pi_0):
        """
        This function carries out the fitting of the natural cubic spline with 3 degrees of freedom of pi_0(lambda).
        The library csaps is used, more can be found at: https://csaps.readthedocs.io/en/latest/tutorial.html
        Note that The smoothing parameter should be in range [0,1], where bounds are:
                0: The smoothing spline is the least-squares straight line fit to the data
                1: The cubic spline interpolant with natural boundary condition

        Hence, to 'choose' the estimate of pi_0 the following is necessary:

                                        pi_0 = fuction(min(lambda))

        Picks out the pi_0 associated with the smallest lambda (threshold) obtained.
        The plot produced shows such fitting.

        Corresponds to step 3. in original paper (spline fitting and estimation).

        Parameters:
        -----------

        LLR: ndarray
            Use output directly from the function approximation_of_LLRs

        pi_0: ndarray
            Use output directly from the function distribution_of_proportion_of_non_osc_cells

        Returns:
        --------

        pi0_guess: float
            Value associated with the minimum lambda after fitting.

        Fig: Fig
            Plot of function obtained, represented as Tuning Parameter (Lambda) Vs Pi_0

        """

        max_LLR = max(max(LLRs[0]), max(LLRs[1]))
        min_LLR = min(min(LLRs[0]), min(LLRs[1]))
        # min_LLR = 0.0 # Temporary fix for -ve LLRs issues.
        min_LLR = max_LLR - 0.9*(max_LLR - min_LLR)
        tuning_parameters = np.linspace(min_LLR, max_LLR, 50)

        splinefit = csaps.CubicSmoothingSpline(tuning_parameters, pi_0, smooth = 0.5)
        x_test = np.linspace(min_LLR, max_LLR, 50)
        y_test = splinefit(x_test)
        results = ((x_test, y_test))
        results = np.vstack(results)

        Fig = plt.figure('Tuning Parameter (Lambda) Vs Pi_0')
        plt.title('Tuning Parameter (Lambda) Vs Pi_0')
        plt.ylabel('Pi0')
        plt.xlabel('Tuning Parameter (Lambda)')
        plt.ylim([0,None])
        plt.xlim([min_LLR - 1, max_LLR + 1])
        plt.scatter(results[0,0], results[1,0], color = 'red')
        plt.plot(tuning_parameters, pi_0, 'o', x_test, y_test, '-')

        pi0_guess = y_test[0]
        # This corresponds to the moment in which, in the paper,
        # we pick the minimum lambda (location [0]) and evaluate our fitted natural cubic spline function
        # which yields the corresponding pi_0.

        return Fig, pi0_guess

    def estimation_of_q_values(self, LLR, pi0_guess):
        '''
        Function to calculate the q-value conditioned on each cell in the dataset.
        By controlling the q-value at a certain threshold (q < γ) we are then able to quantify the
        number of oscillating and non-oscillating cells within the population.

        Parameters:
        -----------

        LLR: ndarray
            Use output directly from the function approximation_of_LLRs

        pi0_guess: float
            Use output directly from the previous function estimation_of_proportion_of_non_osc_cells[1]

        Returns:
        --------

        q_est: ndarray
        Returns a nx1 vector where n is the size of the original observed dataset.
        The chosen cutoff value q-value measures significance based on the False Discovery Rate FDR.
        That is the rate that significant features are truly null. A very small q-value indicates the

        '''

        #Extract LLRs from the vector.
        LLR_data = np.sort(LLR[0])
        LLR_synth = np.sort(LLR[1])
        q_est = np.zeros((len(LLR_data)))

        # First loop deals with the calculation of q-values of each cell in the dataset.
        # In original code threshold t here is the last

        for index, cell in enumerate(LLR_data):
            threshold = cell
            count_data = sum(1 for LLR in LLR_data if LLR >= threshold)
            count_synth = sum(1 for LLR in LLR_synth if LLR >= threshold)
            q_est[index] = pi0_guess*np.divide(np.divide(count_synth,len(LLR_synth)),np.divide(count_data,len(LLR_data)))

        # Second part focuses on the last step and the sentence in the paper:
        # By controlling the q-value at a certain threshold (q < γ) we are then able to quantify the number of oscillating and non-oscillating cells within the population.

        # in paper
        # q = 0.05;%0.05;
        # cutoff = find(q1<q,1,'first')
        # [w,l] = sort(I);
        # Reorderedq = q1(l);
        # PassList = Reorderedq<q;

        # This last part is done in the model_selection function.

        return q_est

    def model_selection(self, observed_timepoints, observed_cells, number_of_synthetic_cells, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.0]):
        """
        Function to run the whole model_selection process.

        Parameters
        ----------

        Same inputs and descriptions as approximation_of_LLRs check if not present here.

        control_cells: ndarray
            A control group is necessary to make accurate inference. You would expect the control cells to be coming from a
            known promoter with constitutive expression (non-oscillatory).
            This array is of size nxm where n is the number of observations and m is the number of control cells.

        observed_cells: ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file (real data) or, if not available,
            from the output of GP.generate_prior_ou_trace().

        control_q_value: float
            Controlling the q value at 0.05 is equivalent to controlling at an FDR of 5%.
            This can be made less stringent, ie 10%, which ideally you would expect it increases the pass rate.

        Returns
        -------



        """

        try:
            number_of_observed_cells = len(observed_cells[0])
        except:
            raise(ValueError("Your number of observed cells must be more than 1 for Model Selection"))

        # Step 1. and 2.
        LLRs = self.approximation_of_LLRs(observed_timepoints, observed_cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess)

        # Optional: Distribution Plot
        DistributionPlot = self.visualisation.LLR_distribution_plot(LLRs[0], LLRs[1])

        # Step 3.
        pi0 = self.distribution_of_proportion_of_non_osc_cells(LLRs, number_of_observed_cells, number_of_synthetic_cells)
        pi0_est = self.estimation_of_proportion_of_non_osc_cells(LLRs, pi0)

        # Step 4.
        q_est = self.estimation_of_q_values(LLRs, pi0_est[1])

        # Optional: Q_values Plot
        QValuesPlot = self.visualisation.q_values_plot(LLRs[0], q_est, control_q_value)

        # Lastly, Inferences:

        #cutoff = np.where(q_est < control_q_value)[0][0]
        reorderedq = np.sort(q_est)
        passlist = reorderedq < control_q_value
        q_values = np.vstack((reorderedq, passlist))
        passed = np.count_nonzero(passlist)
        print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_observed_cells,
              "cells from the data exceed the LLR threshold and are classified as oscillatory")

        return LLRs, q_values, DistributionPlot, QValuesPlot

    def model_selection_with_control(self, observed_timepoints, control_cells, observed_cells, number_of_synthetic_cells, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.001]):
        """
        Function to run the whole model_selection process.

        Parameters
        ----------

        Same inputs and descriptions as approximation_of_LLRs check if not present here.

        control_cells: ndarray
            A control group is necessary to make accurate inference. You would expect the control cells to be coming from a
            known promoter with constitutive expression (non-oscillatory).
            This array is of size nxm where n is the number of observations and m is the number of control cells.

        observed_cells: ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file (real data) or, if not available,
            from the output of GP.generate_prior_ou_trace().

        control_q_value: float
            Controlling the q value at 0.05 is equivalent to controlling at an FDR of 5%.
            This can be made less stringent, ie 10%, which ideally you would expect it increases the pass rate.

        Returns
        -------



        """

        try:
            number_of_observed_cells = len(observed_cells[0])
            number_of_control_cells = len(observed_cells[0])
            cells = np.hstack((observed_cells, control_cells))
        except:
            raise(ValueError("Your number of observed and control cells must be more than 1 for Model Selection"))

        # Step 1. and 2.
        LLRs = self.approximation_of_LLRs(observed_timepoints, cells, number_of_observed_cells + number_of_control_cells, number_of_synthetic_cells, initial_guess)

        # Optional: Distribution Plot
        observed_LLRs = LLRs[0][0:number_of_observed_cells]
        control_LLRs = LLRs[0][number_of_observed_cells:]
        synthetic_LLRs = LLRs[1]
        DistributionPlot = self.visualisation.LLR_distribution_plot_with_control(control_LLRs, observed_LLRs, synthetic_LLRs)

        # Step 3.
        pi0 = self.distribution_of_proportion_of_non_osc_cells(LLRs, number_of_observed_cells + number_of_control_cells, number_of_synthetic_cells)
        pi0_est = self.estimation_of_proportion_of_non_osc_cells(LLRs, pi0)

        # Step 4.
        q_est = self.estimation_of_q_values(LLRs, pi0_est[1])

        # Optional: Q_values Plot
        QValuesPlot = self.visualisation.q_values_plot(LLRs[0], q_est, control_q_value)

        # Lastly, Inferences:

        #cutoff = np.where(q_est < control_q_value)[0][0]
        reorderedq = np.sort(q_est)
        passlist = reorderedq < control_q_value
        q_values = np.vstack((reorderedq, passlist))

        # This is necessary to differentiate q_values estimates as obtained from control vs observed data
        # q_est_observed = q_est[0:number_of_observed_cells]
        # reorderedq = np.sort(q_est_observed)
        # passlist = reorderedq < control_q_value
        passed = np.count_nonzero(passlist)
        print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_observed_cells + number_of_control_cells,
              "cells from the data exceed the LLR threshold and are classified as oscillatory")

        # q_est_control = q_est[number_of_observed_cells:]
        # reorderedq = np.sort(q_est_control)
        # passlist = reorderedq < control_q_value
        # passed = np.count_nonzero(passlist)
        # print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_control_cells,
        #       "cells from the control group exceed the LLR threshold and are classified as oscillatory")

        return LLRs, q_values, DistributionPlot, QValuesPlot

    # NOTE: The following functions are equivalents of the above but taking a different input (ie a list not array)

    def approximation_of_LLRs_for_list(self, observed_timepoints, cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess): #, normalise = True):
        '''
        Choosing a threshold for the LLR relies on a Bootstrap approach to approximate the LLR distribution
        for a population of non-oscillating cells:

        1. Fit OUosc and OU GP, Storing Results and Calculating LLR from each Observed Cell
        2. Simulate Synthetic Cells with the Null Aperiodic OU Model.
        3. Calculate the LLRs for each Synthetic Cell and store them.
            The number of synthetic cells generated is equally
        4. Normalise the LLRs.
        5. Optional: plot a histogram of the LLRs).
            this gives an approximate distribution for the LLRs which allow us to calculate

        Corresponds to step 1. and 2. in original paper.

        Parameters:
        -----------

        observed_timepoints: ndarray
            Input a nx1 array where n is the number of observations per cell. This will represent the timepoints at which
            the gene expression levels for each cell were measured.

        cells : ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file or from the output of GP.generate_prior_ou_trace().

        number_of_observed_cells: integer
            Might not need this as can be retrieved from the cells array but could also leave it as a failsafe that it
            coincides with the number of columns.

        number_of_synth_cells: integer
            As a general rule of thumb, as proposed by 'Identifying stochastic oscillations in single-cell live imaging time
            series using Gaussian processes', it is preferred to pick out at least 20 synthetic cells per observed cell.
            Hence if the number of observed cells is 10, we get a total of 200 synthetic cells to calculate LLRs from.

        normalised: boolean
            In project LLRs from data are normalised before moving forward. Hence default is True.
            Note that in the project only the observed one seem to have been normalised, not the synthetic bootstrap.

        Returns:
        --------

        LLR: ndarray
        Returns an array of size (2,) where
            LLR[0]: represents the normalised LLR values of the time series from the data.
            LLR[1]: represents the LLR values of the synthetic time series from the bootstrapping.
        '''

        start_time = time.time()

        number_of_parameters = 4  # As we have alpha, beta and variance and noise

        par_cell_osc = np.zeros((number_of_parameters, number_of_observed_cells))
        par_cell_non_osc = np.zeros((number_of_parameters, number_of_observed_cells))
        loglik_cell_non_osc = np.zeros((number_of_observed_cells))
        loglik_cell_osc = np.zeros((number_of_observed_cells))
        llikelihood_ratio_observed_cells = np.zeros((number_of_observed_cells))
        normalised_llikelihood_ratio_observed_cells = np.zeros((number_of_observed_cells))

        # First Loop Deals Fitting OUosc and OU GP, Storing Results and Calculating LLR from each Observed Cell

        for cell in range(number_of_observed_cells):

            observed_cell = cells[cell]
            number_of_observations = len(observed_cell)
            observed_timepoints_per_cell = observed_timepoints[0:number_of_observations]

            self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints_per_cell, observed_y = observed_cell)
            op_cell_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess)
            self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints_per_cell, observed_y = observed_cell)
            op_cell_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess)

            par_cell_osc[:,cell] = op_cell_osc.x # Not for computation but for exporting
            par_cell_non_osc[:,cell] = op_cell_non_osc.x # For computation of synthetic cells
            loglik_cell_non_osc[cell] = - op_cell_non_osc.fun
            loglik_cell_osc[cell] = - op_cell_osc.fun
            llikelihood_ratio_observed_cells[cell] = 2*(loglik_cell_osc[cell] - loglik_cell_non_osc[cell])
            # Normalisation as described in project is (LLR/length of data) * 100
            # normalised_llikelihood_ratio_observed_cells[cell] = np.divide(llikelihood_ratio_observed_cells[cell],number_of_observed_cells*number_of_observations)*100

        print("Completed Fitting of", number_of_observed_cells, "Observed Cells")

        # Second Loop Deals with Synthetic Cells Generation to Approximate LLR Distribution
        # Note even if observed cells have different 'end' times, we keep a uniform duration for the synthetic cells.

        number_of_observations = len(observed_timepoints)
        number_of_synth_cell_per_observed_cell = round(number_of_synthetic_cells/number_of_observed_cells)
        # synthetic_data = np.zeros((number_of_observations, number_of_synth_cell_per_observed_cell)) should not be needed
        synthetic_cells = np.zeros((number_of_observed_cells, number_of_observations, number_of_synth_cell_per_observed_cell))
        for cell in range(number_of_observed_cells):

            self.gp = gp.GP(alpha = par_cell_non_osc[0,cell], beta = par_cell_non_osc[1,cell], variance = par_cell_non_osc[2,cell], noise = par_cell_non_osc[3, cell], oscillatory = False)
            synthetic_data = self.gp.generate_prior_ou_trace(duration = observed_timepoints[-1], number_of_observations = number_of_observations, number_of_traces = number_of_synth_cell_per_observed_cell)[:,1:number_of_synth_cell_per_observed_cell+1]
            synthetic_cells[cell, :, :] = np.array([synthetic_data])
            # print("Completed Generation of Synthetic Cells of Observed Cell", cell + 1)

        print("Completed Generation of Synthetic Cells for all", number_of_observed_cells," Observed Cell")

        # Third Loop Deals With Calculating LLR for each new Synthetic Cell
        llikelihood_ratio_synth_cells = np.zeros((number_of_synth_cell_per_observed_cell*number_of_observed_cells))
        i = -1
        j = 0
        for cell in range(number_of_observed_cells):
            for synth_cell in range(number_of_synth_cell_per_observed_cell):
                i = i + 1
                try:
                    self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints, observed_y = synthetic_cells[cell,:,synth_cell])
                    neg_llik_synth_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess).fun
                    self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints, observed_y = synthetic_cells[cell,:,synth_cell])
                    neg_llik_synth_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess).fun
                    llikelihood_ratio_synth_cells[i] = 2*((-neg_llik_synth_osc) - (-neg_llik_synth_non_osc))
                except Exception as e:
                    j = j + 1
                    print(j, "synthetic cells could not be optimised out of the total", number_of_synthetic_cells)
                    print("The Error associated with it is :", e)


            print("Completed Fitting of OUosc and OU GP on Synthetic, Storing Results and Calculating LLR from Observed Cell", cell + 1)

        print("Completed Fitting of OUosc and OU GP on", number_of_synthetic_cells," Synthetic Cells, storing their results and calculating LLRs")

        # Setting negative LLRs to zero or will cause problems down the line.
        llikelihood_ratio_observed_cells[llikelihood_ratio_observed_cells<0] = 0
        normalised_llikelihood_ratio_observed_cells[normalised_llikelihood_ratio_observed_cells<0] = 0
        llikelihood_ratio_synth_cells[llikelihood_ratio_synth_cells<0] = 0

        # Store Results Away
        LLR = np.array((llikelihood_ratio_observed_cells, llikelihood_ratio_synth_cells))
        # LLR = np.array((normalised_llikelihood_ratio_observed_cells, llikelihood_ratio_synth_cells))
        optim_params = np.vstack((par_cell_osc, par_cell_non_osc))

        end_time = time.time()
        print("Execution time:", end_time-start_time)

        return LLR, optim_params

    def model_selection_for_list(self, observed_timepoints, observed_cells, number_of_synthetic_cells, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.0]):
        """
        Function to run the whole model_selection process.

        Parameters
        ----------

        Same inputs and descriptions as approximation_of_LLRs check if not present here.

        control_cells: ndarray
            A control group is necessary to make accurate inference. You would expect the control cells to be coming from a
            known promoter with constitutive expression (non-oscillatory).
            This array is of size nxm where n is the number of observations and m is the number of control cells.

        observed_cells: ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file (real data) or, if not available,
            from the output of GP.generate_prior_ou_trace().

        control_q_value: float
            Controlling the q value at 0.05 is equivalent to controlling at an FDR of 5%.
            This can be made less stringent, ie 10%, which ideally you would expect it increases the pass rate.

        Returns
        -------


        """

        number_of_observed_cells = len(observed_cells)

        # Step 1. and 2.
        approximation_of_LLRs = self.approximation_of_LLRs_for_list(observed_timepoints, observed_cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess)
        optim_parameters = approximation_of_LLRs[1]
        LLRs = approximation_of_LLRs[0]

        # Optional: Distribution Plot
        DistributionPlot = self.visualisation.LLR_distribution_plot(LLRs[0], LLRs[1])

        # Step 3.
        pi0 = self.distribution_of_proportion_of_non_osc_cells(LLRs, number_of_observed_cells, number_of_synthetic_cells)
        pi0_est = self.estimation_of_proportion_of_non_osc_cells(LLRs, pi0)
        Pi0Plot = pi0_est[0]
        pi0_est = pi0_est[1]

        # Step 4.
        q_est = self.estimation_of_q_values(LLRs, pi0_est)

        # Optional: Q_values Plot
        QValuesPlot = self.visualisation.q_values_plot(LLRs[0], q_est, control_q_value)

        # Lastly, Inferences:

        #cutoff = np.where(q_est < control_q_value)[0][0]
        reorderedq = np.sort(q_est)
        passlist = reorderedq < control_q_value
        q_values = np.vstack((reorderedq, passlist))
        passed = np.count_nonzero(passlist)
        print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_observed_cells,
              "cells from the data exceed the LLR threshold and are classified as oscillatory")

        return LLRs, optim_parameters, q_values, DistributionPlot, Pi0Plot, QValuesPlot

    def model_selection_for_list_with_control(self, observed_timepoints, control_cells, observed_cells, number_of_synthetic_cells, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.0]):
        """
        Function to run the whole model_selection process from list with control cells.

        Parameters
        ----------

        Same inputs and descriptions as approximation_of_LLRs check if not present here.

        control_cells: ndarray
            A control group is necessary to make accurate inference. You would expect the control cells to be coming from a
            known promoter with constitutive expression (non-oscillatory).
            This array is of size nxm where n is the number of observations and m is the number of control cells.

        observed_cells: ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file (real data) or, if not available,
            from the output of GP.generate_prior_ou_trace().

        control_q_value: float
            Controlling the q value at 0.05 is equivalent to controlling at an FDR of 5%.
            This can be made less stringent, ie 10%, which ideally you would expect it increases the pass rate.

        Returns
        -------


        """

        number_of_observed_cells = len(observed_cells) + len(control_cells)
        cells = np.hstack((observed_cells, control_cells))

        # Step 1. and 2.
        approximation_of_LLRs = self.approximation_of_LLRs_for_list(observed_timepoints, cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess)
        optim_parameters = approximation_of_LLRs[1]
        LLRs = approximation_of_LLRs[0]

        # Optional: Distribution Plot
        # Optional: Distribution Plot
        observed_LLRs = LLRs[0][0:len(observed_cells)]
        control_LLRs = LLRs[0][len(observed_cells):]
        synthetic_LLRs = LLRs[1]
        DistributionPlot = self.visualisation.LLR_distribution_plot_with_control(control_LLRs, observed_LLRs, synthetic_LLRs)

        # Step 3.
        pi0 = self.distribution_of_proportion_of_non_osc_cells(LLRs, number_of_observed_cells, number_of_synthetic_cells)
        pi0_est = self.estimation_of_proportion_of_non_osc_cells(LLRs, pi0)
        Pi0Plot = pi0_est[0]
        pi0_est = pi0_est[1]

        # Step 4.
        q_est = self.estimation_of_q_values(LLRs, pi0_est)

        # Optional: Q_values Plot
        QValuesPlot = self.visualisation.q_values_plot(LLRs[0], q_est, control_q_value)

        # Lastly, Inferences:

        #cutoff = np.where(q_est < control_q_value)[0][0]
        reorderedq = np.sort(q_est)
        passlist = reorderedq < control_q_value
        q_values = np.vstack((reorderedq, passlist))
        passed = np.count_nonzero(passlist)
        print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_observed_cells,
              "cells from the data exceed the LLR threshold and are classified as oscillatory")

        return LLRs, optim_parameters, q_values, DistributionPlot, Pi0Plot, QValuesPlot

    # Calculating LLR for one cell

    def calculating_LLR(self, observed_timepoints, observed_cell, initial_guess):
        """

        :param observed_timepoints:
        :param observed_cell:
        :param initial_guess:
        :return:
        """

        number_of_observations = len(observed_cell)

        self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints, observed_y = observed_cell)
        op_cell_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess)
        self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints, observed_y = observed_cell)
        op_cell_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess)

        loglik_cell_non_osc = - op_cell_non_osc.fun
        loglik_cell_osc = - op_cell_osc.fun
        llikelihood_ratio = 2*(loglik_cell_osc - loglik_cell_non_osc)
        # Normalisation as described in project is (LLR/length of data) * 100
        # normalised_llikelihood_ratio = np.divide(llikelihood_ratio,number_of_observations)*100

        return llikelihood_ratio

    def contingency_table(self, q_values_observed_cells, q_values_control_cells):
        """
        This can only be run when the control cells have be run as well. Otherwise there will be missing parts.
        Note, this is based on the idea that the control cells are a population of non-oscillating cells and that
        the observed cells are a population of oscillating cells. This is to determine the effectiveness of the model selection.

        Parameters
        ---------

        q_values_observed_cells: ndarray
            Use directly output [2] from model_selection function (called q_values, it includes a passlist)
        q_values_control_cells: ndarray
            Use directly output [2] from model_selection function (called q_values, it includes a passlist)

        Returns
        -------

        contingency_table : ndarray
            As shown in https://en.wikipedia.org/wiki/Receiver_operating_characteristic

        """
        number_of_observed_cells = len(q_values_observed_cells[0])
        number_of_control_cells = len(q_values_control_cells[0])

        passlist_observed = sum(1 for passed in q_values_observed_cells[1] if passed == 1)
        passlist_control = sum(1 for passed in q_values_control_cells[1] if passed == 0)

        true_negatives = passlist_control
        false_negatives = number_of_observed_cells - passlist_observed
        true_positives = passlist_observed
        false_positives = number_of_control_cells - passlist_control

        predicted_condition_positive = np.hstack((true_positives, false_positives))
        predicted_condition_negative = np.hstack((true_negatives, false_negatives))
        contingency_table = np.vstack((predicted_condition_positive, predicted_condition_negative))

        return contingency_table

    def ROC_curves(self, contingency_table):
        """
        When we predict a binary outcome, it is either a correct prediction (true positive) or not (false positive).
        It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis)
        for a number of different candidate threshold values between 0.0 and 1.0.

        Parameters
        --------



        Returns
        --------


        """
        true_positives = contingency_table[0,0]
        false_positives = contingency_table[1,0]
        true_negatives = contingency_table[0,1]
        false_negatives = contingency_table[1,1]

        true_positive_rate = true_positives / (true_positives + false_negatives)
        sensitivity = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)
        specificity = true_negatives / (true_negatives + false_positives)

        # where false_positive_rate = 1 - specificity

        return

    # NOTE: The following functions are equivalent to the model selection for lists above but ensuring that the length of the Synthetic Cells generated is adjusted

    def approximation_of_LLRs_for_list_new(self, observed_timepoints, cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess): #, normalise = True):
        '''
        Choosing a threshold for the LLR relies on a Bootstrap approach to approximate the LLR distribution
        for a population of non-oscillating cells:

        1. Fit OUosc and OU GP, Storing Results and Calculating LLR from each Observed Cell
        2. Simulate Synthetic Cells with the Null Aperiodic OU Model.
        3. Calculate the LLRs for each Synthetic Cell and store them.
            The number of synthetic cells generated is equally
        4. Normalise the LLRs.
        5. Optional: plot a histogram of the LLRs).
            this gives an approximate distribution for the LLRs which allow us to calculate

        Corresponds to step 1. and 2. in original paper.

        Parameters:
        -----------

        observed_timepoints: ndarray
            Input a nx1 array where n is the number of observations per cell. This will represent the timepoints at which
            the gene expression levels for each cell were measured.

        cells : list
            Input a list where each entry is an array representative of an observed cell. The length of 'cells' should be
            equivalent to the number_of_observed_cells.

        number_of_observed_cells: integer
            Might not need this as can be retrieved from the cells array but could also leave it as a failsafe that it
            coincides with the number of columns.

        number_of_synth_cells: integer
            As a general rule of thumb, as proposed by 'Identifying stochastic oscillations in single-cell live imaging time
            series using Gaussian processes', it is preferred to pick out at least 20 synthetic cells per observed cell.
            Hence if the number of observed cells is 10, we get a total of 200 synthetic cells to calculate LLRs from.

        normalised: boolean
            In project LLRs from data are normalised before moving forward. Hence default is True.
            Note that in the project only the observed one seem to have been normalised, not the synthetic bootstrap.

        Returns:
        --------

        LLR: ndarray
        Returns an array of size (2,) where
            LLR[0]: represents the normalised LLR values of the time series from the data.
            LLR[1]: represents the LLR values of the synthetic time series from the bootstrapping.
        '''

        start_time = time.time()

        number_of_parameters = 4  # As we have alpha, beta and variance and noise

        par_cell_osc = np.zeros((number_of_parameters, number_of_observed_cells))
        par_cell_non_osc = np.zeros((number_of_parameters, number_of_observed_cells))
        loglik_cell_non_osc = np.zeros((number_of_observed_cells))
        loglik_cell_osc = np.zeros((number_of_observed_cells))
        llikelihood_ratio_observed_cells = np.zeros((number_of_observed_cells))
        # normalised_llikelihood_ratio_observed_cells = np.zeros((number_of_observed_cells))

        # First Loop Deals Fitting OUosc and OU GP, Storing Results and Calculating LLR from each Observed Cell

        for cell in range(number_of_observed_cells):

            observed_cell = cells[cell]
            number_of_observations_per_cell = len(observed_cell)
            observed_timepoints_per_cell = observed_timepoints[0:number_of_observations_per_cell]

            self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints_per_cell, observed_y = observed_cell)
            op_cell_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess)
            self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints_per_cell, observed_y = observed_cell)
            op_cell_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess)

            par_cell_osc[:,cell] = op_cell_osc.x # Not for computation but for exporting
            par_cell_non_osc[:,cell] = op_cell_non_osc.x # For computation of synthetic cells
            loglik_cell_non_osc[cell] = - op_cell_non_osc.fun
            loglik_cell_osc[cell] = - op_cell_osc.fun
            llikelihood_ratio_observed_cells[cell] = 2*(loglik_cell_osc[cell] - loglik_cell_non_osc[cell])
            # Normalisation as described in project is (LLR/length of data) * 100
            # normalised_llikelihood_ratio_observed_cells[cell] = np.divide(llikelihood_ratio_observed_cells[cell],number_of_observed_cells*number_of_observations)*100

        print("Completed Fitting of", number_of_observed_cells, "Observed Cells")

        # Second Loop Deals with Synthetic Cells Generation to Approximate LLR Distribution
        # Note even if observed cells have different 'end' times, we keep a uniform duration for the synthetic cells.

        number_of_synth_cell_per_observed_cell = round(number_of_synthetic_cells/number_of_observed_cells)
        # synthetic_data = np.zeros((number_of_observations, number_of_synth_cell_per_observed_cell)) should not be needed
        synthetic_cells = []
        for cell in range(number_of_observed_cells):

            observed_cell = cells[cell]
            number_of_observations_per_cell = len(observed_cell)
            observed_timepoints_per_cell = observed_timepoints[0:number_of_observations_per_cell]

            self.gp = gp.GP(alpha = par_cell_non_osc[0,cell], beta = par_cell_non_osc[1,cell], variance = par_cell_non_osc[2,cell], noise = par_cell_non_osc[3, cell], oscillatory = False)
            synthetic_data = self.gp.generate_prior_ou_trace(duration = observed_timepoints_per_cell[-1], number_of_observations = number_of_observations_per_cell, number_of_traces = number_of_synth_cell_per_observed_cell)[:,1:number_of_synth_cell_per_observed_cell+1]
            synthetic_cells.append(synthetic_data) # Store in List, for example entry [1] etc is the ndarray of synthetic traces for cell 1.
            # print("Completed Generation of Synthetic Cells of Observed Cell", cell + 1)

        print("Completed Generation of Synthetic Cells for all", number_of_observed_cells, " Observed Cell")

        # Third Loop Deals With Calculating LLR for each new Synthetic Cell
        llikelihood_ratio_synth_cells = np.zeros((number_of_synth_cell_per_observed_cell*number_of_observed_cells))
        i = -1
        j = 0
        for cell in range(number_of_observed_cells):
            for synth_cell in range(number_of_synth_cell_per_observed_cell):
                i = i + 1
                try:
                    observed_cell = cells[cell]
                    number_of_observations_per_cell = len(observed_cell)
                    observed_timepoints_per_cell = observed_timepoints[0:number_of_observations_per_cell]
                    synthetic_data = synthetic_cells[cell][:,synth_cell]

                    self.optim_non_osc = optimisation.Optimisation(oscillatory = False, observed_timepoints = observed_timepoints_per_cell, observed_y = synthetic_data)
                    neg_llik_synth_non_osc = self.optim_non_osc.optimizing_neg_marginal_loglikelihood(initial_guess).fun
                    self.optim_osc = optimisation.Optimisation(oscillatory = True, observed_timepoints = observed_timepoints_per_cell, observed_y = synthetic_data)
                    neg_llik_synth_osc = self.optim_osc.optimizing_neg_marginal_loglikelihood(initial_guess).fun
                    llikelihood_ratio_synth_cells[i] = 2*((-neg_llik_synth_osc) - (-neg_llik_synth_non_osc))

                except Exception as e:
                    j = j + 1
                    print(j, "synthetic cells could not be optimised out of the total", number_of_synthetic_cells)
                    print("The Error associated with it is :", e)

            print("Completed Fitting of OUosc and OU GP on Synthetic, Storing Results and Calculating LLR from Observed Cell", cell + 1)

        print("Completed Fitting of OUosc and OU GP on", number_of_synthetic_cells," Synthetic Cells, storing their results and calculating LLRs")

        # Setting negative LLRs to zero or will cause problems down the line.
        llikelihood_ratio_observed_cells[llikelihood_ratio_observed_cells<0] = 0
        # normalised_llikelihood_ratio_observed_cells[normalised_llikelihood_ratio_observed_cells<0] = 0
        llikelihood_ratio_synth_cells[llikelihood_ratio_synth_cells<0] = 0

        # Store Results Away
        LLR = np.array((llikelihood_ratio_observed_cells, llikelihood_ratio_synth_cells))
        # LLR = np.array((normalised_llikelihood_ratio_observed_cells, llikelihood_ratio_synth_cells))
        optim_params = np.vstack((par_cell_osc, par_cell_non_osc))

        end_time = time.time()
        print("Execution time:", end_time-start_time)

        return LLR, optim_params

    def model_selection_for_list_new(self, observed_timepoints, observed_cells, number_of_synthetic_cells, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.0]):
        """
        Function to run the whole model_selection process.

        Parameters
        ----------

        Same inputs and descriptions as approximation_of_LLRs check if not present here.

        control_cells: ndarray
            A control group is necessary to make accurate inference. You would expect the control cells to be coming from a
            known promoter with constitutive expression (non-oscillatory).
            This array is of size nxm where n is the number of observations and m is the number of control cells.

        observed_cells: ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file (real data) or, if not available,
            from the output of GP.generate_prior_ou_trace().

        control_q_value: float
            Controlling the q value at 0.05 is equivalent to controlling at an FDR of 5%.
            This can be made less stringent, ie 10%, which ideally you would expect it increases the pass rate.

        Returns
        -------

        """

        number_of_observed_cells = len(observed_cells)

        # Step 1. and 2.
        approximation_of_LLRs = self.approximation_of_LLRs_for_list_new(observed_timepoints, observed_cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess)
        optim_parameters = approximation_of_LLRs[1]
        LLRs = approximation_of_LLRs[0]

        # Optional: Distribution Plot
        DistributionPlot = self.visualisation.LLR_distribution_plot(LLRs[0], LLRs[1])

        # Step 3.
        pi0 = self.distribution_of_proportion_of_non_osc_cells(LLRs, number_of_observed_cells, number_of_synthetic_cells)
        pi0_est = self.estimation_of_proportion_of_non_osc_cells(LLRs, pi0)
        Pi0Plot = pi0_est[0]
        pi0_est = pi0_est[1]

        # Step 4.
        q_est = self.estimation_of_q_values(LLRs, pi0_est)

        # Optional: Q_values Plot
        QValuesPlot = self.visualisation.q_values_plot(LLRs[0], q_est, control_q_value)

        # Lastly, Inferences:

        #cutoff = np.where(q_est < control_q_value)[0][0]
        reorderedq = np.sort(q_est)
        passlist = reorderedq < control_q_value
        q_values = np.vstack((reorderedq, passlist))
        passed = np.count_nonzero(passlist)
        print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_observed_cells,
              "cells from the data exceed the LLR threshold and are classified as oscillatory")

        return LLRs, optim_parameters, q_values, DistributionPlot, Pi0Plot, QValuesPlot

    def model_selection_for_list_new_with_control(self, observed_timepoints, control_cells, observed_cells, number_of_synthetic_cells, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.0]):
        """
        Function to run the whole model_selection process from list with control cells.

        Parameters
        ----------

        Same inputs and descriptions as approximation_of_LLRs check if not present here.

        control_cells: ndarray
            A control group is necessary to make accurate inference. You would expect the control cells to be coming from a
            known promoter with constitutive expression (non-oscillatory).
            This array is of size nxm where n is the number of observations and m is the number of control cells.

        observed_cells: ndarray
            Input a nxm array where n is the number of observations per cell and m is the number of observed cells.
            This can be easily retrieved from an excel file (real data) or, if not available,
            from the output of GP.generate_prior_ou_trace().

        control_q_value: float
            Controlling the q value at 0.05 is equivalent to controlling at an FDR of 5%.
            This can be made less stringent, ie 10%, which ideally you would expect it increases the pass rate.

        Returns
        -------


        """

        number_of_observed_cells = len(observed_cells) + len(control_cells)
        cells = np.hstack((observed_cells, control_cells))

        # Step 1. and 2.
        approximation_of_LLRs = self.approximation_of_LLRs_for_list_new(observed_timepoints, cells, number_of_observed_cells, number_of_synthetic_cells, initial_guess)
        optim_parameters = approximation_of_LLRs[1]
        LLRs = approximation_of_LLRs[0]

        # Optional: Distribution Plot
        observed_LLRs = LLRs[0][0:len(observed_cells)]
        control_LLRs = LLRs[0][len(observed_cells):]
        synthetic_LLRs = LLRs[1]
        DistributionPlot = self.visualisation.LLR_distribution_plot_with_control(control_LLRs, observed_LLRs, synthetic_LLRs)

        # Step 3.
        pi0 = self.distribution_of_proportion_of_non_osc_cells(LLRs, number_of_observed_cells, number_of_synthetic_cells)
        pi0_est = self.estimation_of_proportion_of_non_osc_cells(LLRs, pi0)
        Pi0Plot = pi0_est[0]
        pi0_est = pi0_est[1]

        # Step 4.
        q_est = self.estimation_of_q_values(LLRs, pi0_est)

        # Optional: Q_values Plot
        QValuesPlot = self.visualisation.q_values_plot(LLRs[0], q_est, control_q_value)

        # Lastly, Inferences:

        #cutoff = np.where(q_est < control_q_value)[0][0]
        reorderedq = np.sort(q_est)
        passlist = reorderedq < control_q_value
        q_values = np.vstack((reorderedq, passlist))
        passed = np.count_nonzero(passlist)
        print("With a control q-value of", control_q_value, ",", passed, "out of", number_of_observed_cells,
              "cells from the data exceed the LLR threshold and are classified as oscillatory")

        return LLRs, optim_parameters, q_values, DistributionPlot, Pi0Plot, QValuesPlot






# - - - - - - - - - - - -
# Testing Model Selection, 10 oscillating cells.
#observed_cells = gp.GP(alpha = 0.005, beta = 0.5, variance = 1.0, noise = 0.0, oscillatory = True)
#observed_cells = observed_cells.generate_prior_ou_trace(duration = 100, number_of_observations = 500, number_of_traces = 10)
#observation_id = np.arange(0, 500, 10) #50 observations
#observed_cells = observed_cells[observation_id, :]

#plt.plot(observed_cells[:,0], observed_cells[:,1:11])
#plt.show()

# Including control data, 10 non-oscillating cells.
#control_cells = gp.GP(alpha = 0.2, beta = 0.0, variance = 1.0, noise = 0.0, oscillatory = False)
#control_cells = control_cells.generate_prior_ou_trace(duration = 100, number_of_observations = 500, number_of_traces = 10)
#control_cells = control_cells[observation_id, :]

#plt.plot(control_cells[:,0], control_cells[:,1:11])
#plt.show()

# Merge

#cells = np.hstack((observed_cells, control_cells[:,1:11]))

# Calling Class

#LLR = ModelSelection()
#LLRs = LLR.approximation_of_LLRs(observed_cells[:,0], cells[:,1:21], 20, 200, [0.05, 0.4, 0.5, 0.5])

# 100 cells
# Duration: 314.79759216308594 secs
# Hence ~5.2 mins

# Hence for 200 cells
# I would expect duration to be ~10 minutes

# Hence for 2000 cells
# I would expect duration to be 100 minutes
# 1.4 hours
# which is not too bad

# - - - - - - -

# 200 cells and 20 observed cells
# Duration: 665
# Hence 10 mins
# As expected...

# - - - - - - -


### Running all together
# modelselection = ModelSelection()
# modelselection = modelselection.model_selection(observed_timepoints = observed_cells[:,0], control_cells = control_cells[:,1:11], observed_cells = observed_cells[:,1:11], number_of_synthetic_cells = 200, control_q_value = 0.05, initial_guess = [0.001, 0.0, 0.001, 0.0])

