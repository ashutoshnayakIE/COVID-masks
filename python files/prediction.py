import time
from datetime import datetime
import numpy as np
import pandas as pd
import glob, os
import statsmodels.api as sm
from sklearn import linear_model
import scipy

# predicting growth rates and its bounds
# uses Krinsky-Robb method to predict the bounds
def Krinsky_Robb_Method(res,x_train):
    # this functions shows an example to calculate the bounds around the coefficients
    # calculating covariance and correlation matrix for the the beta coefficients
    # using residual to estimate variance (sse)

    ssquare = np.mean((y_train - res.predict(x_train)) ** 2)
    coefficients = np.array(res.params).reshape(-1, 1)

    # removing the mask residuals (in case control function approach is use)
    # this is because control function approach has multicollinearity and it will lead to unstable covariance matrix.
    if 'Mask_residuals' in list(x_train.columns):
        residual_index = list(x_train.columns).index('Mask_residuals')
        coefficients = np.delete(coefficients, residual_index, 0)
        del x_train['Mask_residuals']

    # calculate covariance matrix
    cov = ssquare * np.linalg.inv(np.matmul(x_train.values.T, x_train.values))

    # cholesky decomposition
    cholesky = np.linalg.cholesky(cov)
    cholesky = np.array(cholesky)

    # adding perturbations to the cholesky matrix
    epsilon = 0.001
    d = len(x_train.columns)
    cholesky = cholesky + epsilon * np.identity(d)

    # generating 10000 samples
    n = 10000
    u = np.random.normal(loc=0, scale=1, size=d * n).reshape(d, n)

    # adding the coefficients (as mean) + generating samples around it by adding standard normal distribution
    beta_samples = coefficients + np.matmul(cholesky, u)

    # confidence intervals can be calculated for these beta samples
    return(beta_samples)

def predicting_growthrate(res, x_train, type_of_function):
    '''
    prediction function can used using the following method
    1. colelct training data from data_shift function
    2. change the values (e.g. masks, mobility, NPI)
    3. use cmobined_effect() function to estimate the combined effect
    4. use the predict function (res.predict(new-x_train) to get the growth rate (new)
    5. use Krinksky-Robb Method to find the confidence interval of the predictions
    '''
    return()

# predicting the total active cases and its bounds
# uses Krinsky-Robb method to predict the bounds
def predicting_infections():
    '''
        prediction function can used using the following method
        1. colelct training data from data_shift function
        2. change the values (e.g. masks, mobility, NPI)
        3. use cmobined_effect() function to estimate the combined effect
        4. use the predict function (res.predict(new-x_train) to get the growth rate (new)
        5. Multuply respective I(t-1) with growth rate to obtain T(t): active infectious cases
        6. use Krinksky-Robb Method to find the confidence interval of the predictions
        '''
    return()