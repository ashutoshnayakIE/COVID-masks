'''
To isolate the effect of masks, we consider a hypothetical country with:
1) 0 fixed effects
2) 0 for mobility (i.e. no change in mobility)
3) 0 NPI (no NPIS included)
4) 0 testing
5) 0 google trend

it shows the effect of masks, as long as everything else remains same
'''

import time
from datetime import datetime
import numpy as np
import pandas as pd
import glob, os
import statsmodels.api as sm
from sklearn import linear_model
import regression_models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC,MultiTaskLassoCV

import scipy
from sklearn.model_selection import KFold

def average_country_results():
    shift = 9
    mob = 'g';
    mobind = [2, 3];
    causal = 0
    days_considered = 60
    type_of_function = 'log'
    x_train, x_train_copy, y_train, x_test, x_test_copy, y_test = data_shift(shift, threshold, mob, mobind, days_considered,
                                                                             causal, type_of_function)

    # we will use res from the linear regression model to obtain the active cases under different scenarios of:
    # 1. mask wearing
    # 2. number of days in the model (present this result in the main result section in the article)

    model = sm.OLS(y_train, x_train)
    res = model.fit()

    # vietnam is selected as it has is the country with no fixed effects in our model
    new_country = x_train_copy[x_train_copy['country'] == 'Vietnam']
    new_country = new_country.sort_values(by=['day'])

    for s in range(len(new_country)):
        # converting all the different variables to 0
        # further, we change only the values of masks to understand and isolate its effect
        for k in mobind:
            new_country['Mobility: ' + m_types[k]].iloc[s] = 0
        new_country['Mask'].iloc[s] = 0
        new_country['Testing'].iloc[s] = 0
        new_country['Trend'].iloc[s] = 0
        for p, pol in enumerate(govtpolicies):
            if pol in new_country.columns:
                new_country[pol].iloc[s] = 0

    # analyzing the results under different values for masks
    # percent refers to the proportion of people wearing face masks in public

    percent = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    average_daily = []
    new_country_copy = pd.DataFrame.copy(new_country)
    del new_country_copy['day']
    del new_country_copy['country']

    # the uniform random draws from standard normal distributions (used in Krinsky - RObb Method)
    d = len(new_data.columns)
    n = 10000
    U = np.random.normal(loc=0, scale=1, size=d * n).reshape(d, n)

    for num, per in enumerate(percent):
        new_data = pd.DataFrame.copy(new_country_copy)

        # we use the 'log' transformation, results can be obtained by using other transformations as well
        new_data['Mask'] = np.log(1 + per)

        actcases = average_bounds(res, new_data, U)

    # actcases can be returned as numpy format of csv format
    # actcases = pd.DataFrame(actcases,columns=['lower bound', 'mean','upper bound')
    return (actcases)



def average_bounds(res, x_train, U):
    # first estimate the covariance matrix
    # U is a multivariate normal random draws (variates)
    ssquare = np.mean((y_train_o - res.predict(x_train_o)) ** 2)
    coefficients = np.array(res.params).reshape(-1, 1)

    cov = ssquare * np.linalg.inv(np.matmul(x_train_o.values.T, x_train_o.values))
    cholesky = np.linalg.cholesky(cov)
    cholesky = np.array(cholesky)

    mask_ind = list(x_train.columns).index('Mask')
    total_ind = [mask_ind]
    mobility_ind = []
    for c, loc in enumerate(list(x_train.columns)):
        if 'Mobility' in loc:
            mobility_ind.append(c)
            total_ind.append(c)

    npi_ind = []
    for c, pol in enumerate(list(x_train.columns)):
        if pol in govtpolicies:
            npi_ind.append(c)
            total_ind.append(c)

    # adding perturbations to the cholesky matrix
    epsilon = 0.001
    d = len(x_train.columns)
    cholesky = cholesky + epsilon * np.identity(d)

    # generating 1000 samples

    np.random.seed(0)
    n = 10000
    beta_samples = coefficients + np.matmul(cholesky, U)

    y_pred = np.matmul(x_train.values, beta_samples)

    # assuming 100 cases to start with
    # however, it does not matter as we present ratio of the active cases with time
    # as compared to the number of cases with 0 mask wearing

    newCumCases = np.ones((61, n)) * 100

    # finding the new cases for the 10000 random draws
    for nn in range(n):
        for s in range(1, len(x_train)):
            newCumCases[s, nn] = newCumCases[s - 1, nn] * np.exp(y_pred[s, nn])
            if newCumCases[s, nn] == 0:
                print(newCumCases[s - 1, nn], np.exp(y_pred[s, nn]))

    actcases = np.zeros((61, 3))

    for s in range(1, len(x_train)):
        actcases[s, 0], actcases[s, 1], actcases[s, 2] = regression_models.CI(newCumCases[s, :])

    return (actcases)