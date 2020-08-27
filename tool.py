# Tool constructed to run the code, calling functions from the classes as defined in BioGP/classes

# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
import os
import classes.data_prep as prep
import classes.gp as gp
import classes.data_visualisation as visualisation
import classes.optimisation as optimisation
import classes.gillespie as gillespie
import classes.model_selection as model_selection
from matplotlib.backends.backend_pdf import PdfPages

#### Step 1. Data Input ---> uses class Data_Input()

print(" - - - Data Input Stage.")
print(" - - - Please input path to excel file containing the observed cells to begin - - - ")
print(" - - - Required: check the necessary formatting of the Excel File as detailed the docs - - - ")

df = prep.Data_Input().get_excel()

print(" - - - How many cells have you observed? - - - ")
number_of_observed_cells = int(input("Input Integer Value : "))
print(" - - - Do you wish to include a set of control cells? - - -")
control_cells_answer = input('yes or no? ')

if control_cells_answer == 'yes':
    print(" - - - Please input path to excel file containing the control cells to continue - - - ")
    df_control = prep.Data_Input().get_excel()
    print(" - - - How many control cells have you observed? - - - ")
    number_of_control_cells = int(input("Input Integer Value : "))

# - -

# Removing NaNs and storing cells (different dimensions) in a list of arrays
observed_cells = prep.Data_Input().remove_nans(number_of_observed_cells, df)
if control_cells_answer == 'yes':
    control_cells = prep.Data_Input().remove_nans(number_of_control_cells, df_control)

#### Step 2. Data Prep ---> uses class Normalisation() and Detrending()

print(" - - - Data Prep Stage.")
print(" - - - Do the observed timepoints need to be converted from minutes into hours? - - - ")
print(" - - - Note: all calculations are calculated in hours.")
time_conversion_answer = input("yes or no? ")

time = df.iloc[:,0]
time = time.to_numpy()[2:len(time)]     #Store Time into a numpy array. #todo: change 2 to 1 when formatting convention has been finalised.

if time_conversion_answer == 'yes':
    time = time/(60*60*1000)                #Convert from ms to hours

# Optional Plot - Data Input
Plot1 = plt.figure("Observed Data Input")
for cell in range(number_of_observed_cells):
    plt.plot(time[0:len(observed_cells[cell])], observed_cells[cell])
plt.title(" Observed Data Input ")
plt.xlabel(" Time (hours) ")
plt.ylabel(" Gene Expression Level ")
plt.show(block = False)

# - -

print(" - - - Does the data need to be normalised? - - - ")
norm_answer = input("yes or no? ")

if norm_answer == 'yes':
    normalised_cells = prep.Normalisation().normalise_many_cells_from_list(observed_cells)
    # Optional Plot - Normalised Cells
    Plot1 = plt.figure("Normalised Observed Data")
    for cell in range(number_of_observed_cells):
        plt.plot(time[0:len(observed_cells[cell])], normalised_cells[cell])
    plt.title("Normalised Data Input")
    plt.xlabel(" Time (hours) ")
    plt.ylabel(" Gene Expression Level ")
    plt.show(block = False)

    if control_cells_answer == 'yes':
        normalised_control_cells = prep.Normalisation().normalise_many_cells_from_list(control_cells)

# - -

print(" - - - Does the data need to be detrended? - - - ")
detrending_answer = input("yes or no? ")

if detrending_answer == 'yes':
    print(" - - - Which detrending parameter do you wish to use? - - - ")
    print(" - - - Recommended: 0.0001")
    print(" - - - Note: if unsure about the detrending process using Gaussian Processes check out the docs.")
    detrending_parameter = float(input("Input Detrending Parameter Alpha : "))
    detrended_data = prep.Detrending(alpha = detrending_parameter).detrend_data_from_list(time, normalised_cells)[0]

    # Optional Plot - Detrended Cells
    Plot2 = plt.figure("Detrended Observed Data")
    for cell in range(number_of_observed_cells):
        plt.plot(time[0:len(observed_cells[cell])], detrended_data[cell])
    plt.title(" Detrended Observed Data")
    plt.xlabel(" Time (hours) ")
    plt.ylabel(" Gene Expression Level ")
    plt.show(block = True)

    if control_cells_answer == 'yes':
        detrended_control_data = prep.Detrending(alpha = detrending_parameter).detrend_data_from_list(time, normalised_control_cells)[0]

# - -

#### Step 3. Single Cell Data Analysis

print(" - - - Do you wish to carry out a single cell data analysis? - - - ")
print(" - - - This will allow you to visualise the fitting of Gaussian Processes - - - ")
single_cell_answer = input("yes or no? ")

if single_cell_answer == 'yes':
    print(" - - - Perfect! Note that the first of the observed cells will be used for this section - - - ")
    print(" - - - Fitting of the Oscillatory and Non-Oscillatory Gaussian Process Models to a single cell - - -")
    print(" - - - Which hyper parameter values for the fitting do you wish to use? - - - ")
    alpha_fit = float(input("Input Fitting Value Alpha : "))
    beta_fit = float(input("Input Fitting Value Beta : "))
    variance_fit = float(input("Input Fitting Value Variance : "))
    noise_fit = float(input("Input Fitting Value Noise : "))
    observed_cell = detrended_data[0]
    observed_timepoints_of_cell = time[0:len(observed_cell)]
    fit_OUosc = visualisation.Visualisation_GP(alpha_fit, beta_fit, variance_fit, noise_fit, True, observed_timepoints_of_cell, observed_cell)
    fit_OUosc = fit_OUosc.gp_ou_trace_3subplot(observed_timepoints_of_cell[-1], 500, 2)
    plt.show(block = False)
    fit_OU = visualisation.Visualisation_GP(alpha_fit, 0.0, variance_fit, noise_fit, False, observed_timepoints_of_cell, observed_cell)
    fit_OU = fit_OU.gp_ou_trace_3subplot(observed_timepoints_of_cell[-1], 500, 2)
    plt.show(block = True)

    print(" - - - Optimisation of Hyper Parameters and Refitting of the Models to the Cell")
    print(" - - - Note: the previous hyper parameter inputs will be used as starting values")
    optim = optimisation.Optimisation(True, observed_timepoints_of_cell, observed_cell)
    optim = optim.optimizing_neg_marginal_loglikelihood([alpha_fit, beta_fit, variance_fit, noise_fit])
    print("OUosc - The optimised hyperparameter alpha:", optim.x[0])
    print("OUosc - The optimised hyperparameter beta:", optim.x[1])
    print("OUosc - The optimised hyperparameter variance:", optim.x[2])
    print("OUosc - The optimised hyperparameter noise:", optim.x[3])
    fit_OUosc = visualisation.Visualisation_GP(optim.x[0], optim.x[1], optim.x[2], optim.x[3], True, observed_timepoints_of_cell, observed_cell)
    fit_OUosc = fit_OUosc.gp_ou_trace_3subplot(observed_timepoints_of_cell[-1], 500, 2)
    visualisation.Visualisation_Optimiser(True, observed_timepoints_of_cell, observed_cell, optim)
    plt.show(block = True)
    optim = optimisation.Optimisation(False, observed_timepoints_of_cell, observed_cell)
    optim = optim.optimizing_neg_marginal_loglikelihood([alpha_fit, beta_fit, variance_fit, noise_fit])
    print("OU - The optimised hyperparameter alpha:", optim.x[0])
    print("OU - The optimised hyperparameter variance:", optim.x[2])
    print("OU - The optimised hyperparameter noise:", optim.x[3])
    fit_OUosc = visualisation.Visualisation_GP(optim.x[0], optim.x[1], optim.x[2], optim.x[3], False, observed_timepoints_of_cell, observed_cell)
    fit_OUosc = fit_OUosc.gp_ou_trace_3subplot(observed_timepoints_of_cell[-1], 500, 2)
    visualisation.Visualisation_Optimiser(False, observed_timepoints_of_cell, observed_cell, optim)
    plt.show(block = True)

#### Step 4. Data Analysis

print(" - - - Determining whether cells are Oscillatory or Non-Oscillatory - - - ")
print(" - - - Recommended: check out the docs on the specifics of the following inputs.")
print(" - - - How many synthetic cells do you wish to generate? - - -")
print(" - - - Note: a large number of observations (>100) and a large number of synthetic cells (>500) can make the model selection 1+ hours.")
print(" - - - Recommended: check out the docs for more information on the model selection execution time expectations.")
synthetic_cells_answer = int(input(" Input Integer Value : "))
print(" - - - Which control q value do you wish to use? - - - ")
control_q_value_answer = float(input(" Input Float Value : "))
print(" - - - Which start values for the optimisation do you wish to use? - - - ")
alpha_start = input("Input Optimisation Starting Value Alpha : ")
beta_start = input("Input Optimisation Starting Value Beta : ")
variance_start = input("Input Optimisation Starting Value Variance : ")
noise_start = input("Input Optimisation Starting Value Noise : ")

if control_cells_answer != 'yes':
    modelselection = model_selection.ModelSelection().model_selection_for_list_new(observed_timepoints = time, observed_cells = detrended_data, number_of_synthetic_cells = synthetic_cells_answer, control_q_value = control_q_value_answer, initial_guess = [alpha_start, beta_start, variance_start, noise_start])
else:
    modelselection = model_selection.ModelSelection().model_selection_for_list_new_with_control(observed_timepoints = time, control_cells = detrended_control_data, observed_cells = detrended_data, number_of_synthetic_cells = synthetic_cells_answer, control_q_value = control_q_value_answer, initial_guess = [alpha_start, beta_start, variance_start, noise_start])

    # print(" - - - First, observed cells execution of model selection - - -")
    # modelselection_observed = model_selection.ModelSelection().model_selection_for_list(observed_timepoints = time, observed_cells = detrended_data, number_of_synthetic_cells = synthetic_cells_answer, control_q_value = control_q_value_answer, initial_guess = [alpha_start, beta_start, variance_start, noise_start])
    # print(" - - - Observed Cells model selection completed successfully! - - -")
    # print(" - - - Secondly, control cells execution of model selection - - -")
    # modelselection_control = model_selection.ModelSelection().model_selection_for_list(observed_timepoints = time, observed_cells = detrended_control_data, number_of_synthetic_cells = synthetic_cells_answer, control_q_value = control_q_value_answer, initial_guess = [alpha_start, beta_start, variance_start, noise_start])
    # print(" - - - Control Cells model selection completed successfully! - - -")

    # q_values_observed = modelselection_observed[2]
    # q_values_control = modelselection_control[2]

    # contingency_table = model_selection.ModelSelection().contingency_table(q_values_observed, q_values_control)
    # print(" - - - Contingency Table - - - ")
    # print(contingency_table)

#### Step 4. Data Export

# Outputs of model_selection: LLRs, optim_params, q_values, DistributionPlot, QValuesPlot

print(" - - - Do you wish to export resulting data for further independent analysis? - - - ")
export_answer = input("yes or no? ")
print(" - - - Should Model Selection plots be included? - - - ")
export_plot_answer = input("yes or no? ")

if export_answer == 'yes':
    my_path = os.path.dirname(os.path.realpath('__file__')) # Figures out the absolute path for you in case your working directory moves around.

    if control_cells_answer == 'yes':

        if export_plot_answer == 'yes':
            modelselection[3].savefig(os.path.join(my_path, "results/Distribution of LLRs.png"))
            modelselection[4].savefig(os.path.join(my_path, "results/Pi0 Distribution Plot.png"))
            modelselection[5].savefig(os.path.join(my_path, "results/Q Values Plot.png"))
            # modelselection_control[3].savefig(os.path.join(my_path, "Results/Control Distribution of LLRs.png"))
            # modelselection_control[4].savefig(os.path.join(my_path, "Results/Control Pi0 Distribution Plot.png"))
            # modelselection_control[5].savefig(os.path.join(my_path, "Results/Control Q Values Plot.png"))

        LLRs_observed = modelselection[0]
        parameters_observed = modelselection[1]
        q_values = modelselection[2]

        # LLRs_observed = modelselection_observed[0]
        # parameters_observed = modelselection_observed[1]

        export_observed_detrended_data = prep.Data_Export().export_detrended_data(detrended_data)
        export_observed_parameters = prep.Data_Export().export_parameter_estimates(parameters_observed)
        export_observed_LLRs = prep.Data_Export().export_LLRs_estimates(LLRs_observed)
        export_observed_period = prep.Data_Export().period_estimation(parameters_observed)
        export_q_values = prep.Data_Export().export_q_values(q_values)
        # export_contingency_table = prep.Data_Export().export_contingency_table(contingency_table)

        wb = Workbook()
        wb.save(os.path.join(my_path, "Results/results.xlsx"))

        with pd.ExcelWriter(os.path.join(my_path, "results/results.xlsx")) as writer:
            export_observed_detrended_data.to_excel(writer, sheet_name = 'Detrended Data')
            export_observed_parameters.to_excel(writer, sheet_name = 'Estimated Parameters')
            export_observed_LLRs.to_excel(writer, sheet_name = 'LLR Distributions')
            export_observed_period.to_excel(writer, sheet_name = 'Period Estimation')
            export_q_values.to_excel(writer, sheet_name = 'Q Values')
            # export_contingency_table.to_excel(writer, sheet_name = 'Contingency Table')
    else:

        if export_plot_answer == 'yes':
            modelselection[3].savefig(os.path.join(my_path, "results/Distribution of LLRs.png"))
            modelselection[4].savefig(os.path.join(my_path, "results/Pi0 Distribution Plot.png"))
            modelselection[5].savefig(os.path.join(my_path, "results/Q Values Plot.png"))

        LLRs = modelselection[0]
        parameters = modelselection[1]

        export_detrended_data = prep.Data_Export().export_detrended_data(detrended_data)
        export_parameters = prep.Data_Export().export_parameter_estimates(parameters)
        export_LLRs = prep.Data_Export().export_LLRs_estimates(LLRs)
        export_period = prep.Data_Export().period_estimation(parameters)

        wb = Workbook()
        wb.save(os.path.join(my_path, "results/results.xlsx"))

        with pd.ExcelWriter(os.path.join(my_path, "results/results.xlsx")) as writer:
            export_detrended_data.to_excel(writer, sheet_name = 'Detrended Data')
            export_parameters.to_excel(writer, sheet_name = 'Estimated Parameters')
            export_LLRs.to_excel(writer, sheet_name = 'LLR Distributions')
            export_period.to_excel(writer, sheet_name = 'Period Estimation')


