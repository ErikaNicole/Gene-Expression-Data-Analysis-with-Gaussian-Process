# 1. Import

import sys
sys.path.append('../') # Necessary to access classes functions from other folder

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv as cv
import classes.data_prep as prep
import classes.gp as gp
import classes.data_visualisation as visualisation
import classes.optimisation as optimisation
import classes.model_selection as model_selection
from matplotlib.backends.backend_pdf import PdfPages

import warnings # Necessary to avoid certain common warnings from coming up. Won't affect inference.
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')

my_path = os.getcwd()
# my_path = os.path.dirname(os.path.realpath('__file__'))

# 2. Import Data from Excel and Prep it

# Collecting real data from an excel spreadsheet is made easy thanks to the gel_excel() function in the data_prep class.
# Similary, to prep the collected data we need to convert time to hours, normalise the data and detrend it.
# The choice of the detrending parameter is

print("Observed Cells Dataset")
df = prep.Data_Input().get_excel()
print("Control Cells Dataset")
df_control = prep.Data_Input().get_excel()
number_of_observed_cells = 19
number_of_control_cells = 25

# Check no NaNs are present/remove if so, then store in list.

observed_cells = prep.Data_Input().remove_nans(number_of_observed_cells, df)
control_cells = prep.Data_Input().remove_nans(number_of_control_cells, df_control)

# Time in Hours

times = df.iloc[:,0]
times = times.to_numpy()[2:len(times)]
times = times/(60*60*1000)

control_times = df_control.iloc[:,0]
control_times = control_times.to_numpy()[2:len(control_times)]
control_times = control_times/(60*60*1000)

# Normalisation

normalised_cells = prep.Normalisation().normalise_many_cells_from_list(observed_cells)
normalised_control_cells = prep.Normalisation().normalise_many_cells_from_list(control_cells)

# Detrending

detrending_parameter_initial_value = np.exp(-2)
detrended_data = prep.Detrending(alpha = detrending_parameter_initial_value).detrend_data_from_list(times, normalised_cells)
detrended_control_data = prep.Detrending(alpha = detrending_parameter_initial_value).detrend_data_from_list(control_times, normalised_control_cells)[0]

# - - Plot of one Detrended Traces against the Original Normalised Signal

trendfit_x = detrended_data[1]
trendfit_y_cell = detrended_data[2][0]
detrended_cell = detrended_data[0][0]
detrended_data = detrended_data[0]

Fig = plt.figure("Detrended Data", figsize = (10, 6))
plt.xlabel("Time (Hours)")
plt.ylabel("Gene Expression")
plt.plot(times[0:len(detrended_cell)], detrended_cell, label = "detrended cell")
plt.plot(times[0:len(detrended_cell)], normalised_cells[0], label = "normalised cell")
plt.plot(trendfit_x, trendfit_y_cell, ls = '--', color = "black", label = "SE fit")
plt.legend()
plt.xlim(0, times[0:len(detrended_cell)][-1])
plt.ylim(-3, 4)
Fig.savefig(os.path.join(my_path, "results/Example_of_a_Detrended Cell.pdf"))

print("Detrending Completed")

# 3. Model Selection Process for all observed and control cells

# This model selection relies on the bootstrap approximation of the distribution of LLRs for a population of non-oscillating cells.
# This is then used as a frame of reference to compare our observed LLR distributions to identify both visually and analytically whether
# our sample of observed cells are behaving in an oscillatory or non-oscillatory manner. Analytically the estimation of q-values is used.
# For more information on the Model Selection process in detail I recommend checking out the documentation.

# Warning: model selection is currently the most expensive in terms of running time.

print("Starting Model Selection...")

modelselection = model_selection.ModelSelection()
modelselection_obs = modelselection.model_selection_for_list_new(observed_timepoints = times, observed_cells = detrended_data, number_of_synthetic_cells = 2000, control_q_value = 0.05, initial_guess = [0.001, 0.5, 0.5, 0.5])

# Rename plot;
os.rename(os.path.join(my_path, "results/LLR_Distribution_Plot.pdf"), os.path.join(my_path, "results/LLR_Distribution_Plot_(Observed Group).pdf"))

modelselection_control = modelselection.model_selection_for_list_new(observed_timepoints = control_times, observed_cells = detrended_control_data, number_of_synthetic_cells = 2000, control_q_value = 0.05, initial_guess = [0.001, 0.5, 0.5, 0.5])

# Rename plot;
os.rename(os.path.join(my_path, "results/LLR_Distribution_Plot.pdf"), os.path.join(my_path, "results/LLR_Distribution_Plot_(Control Group).pdf"))

print("... Success!!!!!")
