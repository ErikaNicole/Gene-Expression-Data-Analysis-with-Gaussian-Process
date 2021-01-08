Doc covering the de-trending step intuition.

## Table of Contents

1. **What is De-Trending?**
2. **Method**
3. **The De-Trending Step Overview**
4. **Functions Overview**
5. **Credits**

## What is De-Trending? <a name="What is De-Trending"></a>

De-trending in most application is the process of removing a certain trend from a time series.
The choice of trend to be taken away is often dependent on the reasons for which the time series
is being studied in the first place.

In our application area, we are studying time series which have some short term oscillatory behaviour
we want to examine. On top of this short term oscillatory behaviour we have a long term trend which may
distort our interpretation of the short term oscillatory signal results, hence we want to remove such trend.

Therefore, the de-trending process in this project aims to identify the long term trends exhibited
by the time series, whilst retaining the short term oscillatory behaviour. 

## Method

Trends tend to be most often represented as a polynomial and choosing the right functional form is often very
specific to the time series analysed. To avoid having to make such a choice, the method deployed here is one
which uses **Gaussian Processes Regression** using a **Squared Exponential (SE) covariate function**.

This ensures we do not have to make a choice of the functional form and can be therefore applied to time series which behave
fairly differently and that may have different long term trends behaviours.

The SE covariate function is defined as follows

![SE_cov_fun](https://github.com/ErikaNicole/Gene-Expression-Data-Analysis-with-Gaussian-Process/docs/markdown_imag/SE_cov_fun_png.png)

![SE_cov_fun](markdown_imag/SE_cov_fun_png.png)

<img src="markdown_imag/SE_cov_fun_png.png">

The most useful property of this particular covariate function is that it is
**infinitely differentiable** ([1][1]), hence very smooth. This enables a smooth fitting to our time series
via the GP Regression, which conveniently resembles the long term trend we aim to identify.

The **hyperparameters** of interest here are:

- ###### alpha_se (todo: latex rendering)

    alpha_se may also be represented as `l = sqrt(1/2*alpha_se)` defining l as the characteristic length-scale of correlation.
    The length-scale controls the frequency of changes within our interval.
    
    For small length-scale you would expect a high frequency of changes in the interval, which may be equivalently seen as 
    function values changing quickly. Alternatively, for large length-scale you would expect the function values to change
    slowly. This has an inverse relationship with alpha_se as shown by their relationship. 
    
- ###### variance_se (todo: latex rendering)

    signal variance is a scaling factor, it determines variation of function values from their mean.
    Small variance_se means the function will stay close to its mean value, larger values allow for a larger departure
    from its mean value. 
    
- ###### noise_se (todo: latex rendering)
    
    the noise parameter is not explicitly part of the SE function, however it is included when calculating the
    Gaussian Process. This represents the variability of the training data, therefore changing the fitted line's
    distance from the training set data points. 

_Intuitively_, since we want to fit long term trends, we do not want our fitted function to have large variability
in the time interval, as this would be equivalently expressed visually by a large number of fluctuations.
Hence, this indicates that **in the de-trending step we wish to keep the parameter alpha_se quite small,
to represent a process which fluctuates slowly over time**. This will be imposed as a condition in the upper boundary
of the optimisation step.

[1]: http://www.gaussianprocess.org/gpml/chapters/RW4.pdf

## The De-Trending Step Overview

This is equivalent to a step-by-step description of the function `detrended_data()` which calls upon all
other functions in this class to carry out the de-trending step.

1. Requires some starting values for our parameter set [alpha_se, variance_se, noise_se], these will be used
 to initialise the optimisation step, as well as the training set of values.
2. A Gaussian Process using a SE covariance function is used to fit the training data provided.
To carry this out:

    _a)_ For a total n time series in training data set, **optimise the hyper parameters** of each time series individually.
        This is done using the function optimise_SE_trace().
        
    _b)_ Store each optimised set of hyper parameters and use these to fit the corresponding time series using fit_SE().
        This will **fit using the Gaussian Process reliant on the Squared Exponential covariate function**. 
        Note: when fitting we fit the model at more data points than the observed number of data points. This is to
        ensure a certain smoothness when plotted.  
        
    _c)_ Having stored the fitted trend for each time series, we can match up the x values of the two
        and **take away the trend from the time series it originated from**.
        Hence, we store the de-trended time series to be returned.
        
3. Returns de-trended time series, their respective fitted trends,
    and the set of time points used to evaluate the fitted trends.

## Functions Overview

Under the class **Detrending()** we have the following functions:

- __init__(self, alpha, variance, noise):

        ''' Set up the initial starting value parameters for the de-trending fitting. '''
    
- **cov_matrix_SE**(self, x1, x2):
        
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

- **generate_prior_SE_trace**(self, test_timepoints, number_of_traces = 1):
        
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
            This array has shape (number_of_observations, number_of_traces + 1)
            where, the first column contains time points,
                   the other columns contains function values of the SE process.
        '''
- **optimise_SE_trace**(self, start_alpha, start_variance, start_noise, observed_timepoints, observed_y):

        """ Use starting values from __init__ to optimise the hyper parameters: [alpha_se, variance_se, noise_se] for 
            the GP fitting. Optimisation is done by minimising the negative marginal log likelihood with respect to the
            Hyper Parameters of the SE function matrix.
        
            The following bounds on the optimisation routine are used:
                
                [alpha_se, variance_se, noise_se] = [(1e-10, exp(-4), (1e-10, None), (1e-10, None)]
                
            This implies variance_se and noise_se are bound to be always positive, whilst alpha_se is bound to be between
            zero and exp(-4). The choice is made and explained in the docs.
            In short: we want to detect long term trends, hence we want a fitted function which varies slowly over the interval.
            Such that we don't mistakenly remove some of the short term oscillations. 
            
        Parameters:
        ------------
        
            start_alpha:
                optimisation starting value for alpha_se
            start_variance:
                optimisation starting value for variance_se
            start_noise:
                optimisation starting value for noise_se
            observed_timepoints: 
                The vector of training inputs which have been observed.
                It takes vector inputs as a number of observations are needed to train the model well.
                Size of the vector |x| = N.
            observed_y:
                The vector of training inputs which have been observed from a single time series.
                Size of the vector |y| = N.
        
        Return:
        ----------
            
            optim.x : ndarray 
                In order, the optimised hyperparameters returned are [alpha_se, variance_se, noise_se].
                
        """

- **fit_SE**(self, observed_timepoints, observed_y, test_timepoints, number_of_traces = 1, cholesky_decompose = True):

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
- **detrend_data**(self, observed_timepoints, data_to_detrend, cholesky_decompose = True):
        
        '''
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

        '''
- **detrend_data_from_list** function is same as above but data_to_detrend takes _list input instead of array input_.
    The reason is that often real data may have time series of different lengths.
    _For example_, in vitro, certain cells may die before others, so the observed measurements would have different
    durations. 

## Credits

The steps described here rely entirely on the research article;

[_Identifying stochastic oscillations in single-cell live imaging time series using Gaussian processes_][2]

Nick E. Phillips, Cerys Manning, Nancy Papalopulu, Magnus Rattray

_Faculty of Biology, Medicine and Health, University of Manchester, Manchester, United Kingdom_

[2]: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005479

