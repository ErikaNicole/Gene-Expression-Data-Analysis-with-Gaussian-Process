# Gene Expression Data Analysis with Gaussian Process
 
### Welcome to the repository!

## The Context

This repository can be used to carry out the analysis of **single cell gene expression levels data**.
By analysis it is intended the study and identification of biological oscillations from a gene network.
Ultimately the tool aims to answer question:

> Is there valid reason to believe the gene expression levels of a cell are showing an oscillatory pattern?

Oscillatory dynamics are often at the forefront of current research due to the correlation to cellular functions.
They may serve as measures of time and space, as well as may have the ability of encoding information.

_For more information_ on the biological context of oscillatory patterns and why they are of interest in biology,
you may want to check out the following sources: **insert links here**

## References

This repository contains a documented and **fully pythonic tool** based on the paper:

 - Phillips NE, Manning C, Papalopulu N, Rattray M.
    (2017) Identifying stochastic oscillations in single-cell live imaging time series using Gaussian processes.
    
    PLoS Comput Biol 13(5): e1005479. https://doi.org/10.1371/journal. pcbi.1005479

## The Method

At the root of studying the time evolution of gene expression levels, lies the task of modeling stochastic time series data.
This is why the method applied relies mainly on the **application of Gaussian Process**.
This is a particularly convenient tool in many application areas such as finance, biology and more,
and is actively being used in current research.

In this specific application, it can help shed some light in **quantifying the proportion of oscillating cells** within 
a population, as well as giving measures such as the **period and quality of oscillations**.

_For more information_ on the application of Gaussian Processes, you may want to look at a detailed description of the
method used here in the document `TheMethod.md`

## The File Structure

The current release of the package is intended for the use of biologists using real collected data from single cell experiments.

#### Under the `classes` folder you will find;

 - `data_prep.py`
 
   This python file contains the classes Data_Input(), Normalisation(), Detrending() and Data_Export().
   The classes name should give you an intuitive understanding of their functionalities. They conveniently take care 
   of all the required initial steps to set up data in the required format for the Gaussian Process Oscillatory and Non-Oscillatory Model Fittings.
   
 - `gp.py`
 
    This python file contains the class GP(). This very small class which contains the 3 essential functions involved with the
    Gaussian Process models fitting.
    For this GP we use _two covariance functions_, one for the non-oscillatory model fitting and one for the oscillatory model.
    Both are contained by the function _cov_matrix_ou_ and are accessed by specifying Oscillatory = 'True' or 'False' in the input.
    
    The other functions are responsible for providing access to the Prior Traces, Posterior Traces as well as the Mean Predictor. 
    
 - `optimisation.py`
 
    This python file contains the class Optimisation(). This focuses on the minimisation of the negative log marginal likelihood
    to output the optimal hyperparameter estimates to use in the Gaussian Process fittings.
    
 - `model_selection.py`
 
    This python file contains the class ModelSelection(). This focuses on the application of the model selection process to identify
    a population of cells oscillatory or non-oscillatory behaviour. It relies on a bootstrap approach to approximate the distribution of LLRs
    for a population of non-oscillating cells and comparing to that the observed LLR distribution.
    
    _For more information_ on the model selection process steps, check out the document `TheMethod.md`
    
 - `data_visualisation.py`
 
    This python file contains the classes Visualisation_GP(), Visualisation_Optimiser(), Visualisation_ModelSelection() and Visualisation_DataPrep().
    This indicates that each of the previous classes have some associated plotting function.
    
    Visualisation_GP() takes care of plotting any number of prior traces and/or posterior traces for
    a choice of hyperparameters and a set of observed data points.
    These plots can be accessed by the user if wished, check out the `Tutorial` on how to.

    Visualisation_Optimiser() takes care of plotting hyperparameter densities with respect to the log marginal likelihood.
    These plots can be accessed by the user if wished, check out the `Tutorial` on how to.

    Visualisation_ModelSelection() takes care of plotting LLR distributions as well as a q_value plot.
    These plots do not need to be accessed by the user as the model_selection class will automatically call onto them when running.
    
    
#### Under the `data` folder you will find;

 - Hes1 Promoter.xlsx and Control Promoter.xlsx
   
   These two are real datasets representing a control promoter and a Hes1 promoter which has previously been reported to oscillate.
   
   
#### Under the `tutorial` folder you will find;

 - `tutorial.ipynb`
 
    This is a walk-through of every single functionality of the classes as laid out in the current release.
    **This is convenient for whoever wants to carry out extra analysis in python directly**.
 
#### In the main repository you will find;

 - `tool.py`
 
   This python file is a completely **hands-free way of running the whole process**. It will ask the user to input
   some information with regards to the data and what they wish to do with it, but will require no coding at all.
   It can be run from terminal, or from a Python shell, and will save the plots and results from the model selection in
   an excel file in a folder to access for independent analysis.
   **This is convenient for whoever doesn't want to carry out extra analysis in python.**

#### Under the `results` folder you will find;

 The excel file and plots from the running of tool.py
 

## The Dependencies

 - numpy
 - matplotlib
 - pandas
 - scipy.optimize
 - scipy.linalg
 - time
 - csaps
 - functools
 - openpyxl
 - os 

## Running the Code

There are two options:

1. Run the `tool.py` function in your terminal or Python shell.
   You will guided through the necessary steps for .
    
2. Take control of the package in python and call onto the functions you wish to use.
   For some guidance check out the `Tutorial.ipynb`

## Contact Info

For any questions and queries regarding this repository feel free to email Erika Gravina at

`eng3@st-andrews.ac.uk`
