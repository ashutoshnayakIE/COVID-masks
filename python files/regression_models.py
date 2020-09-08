import time
from datetime import datetime
import numpy as np
import pandas as pd
import glob, os
import statsmodels.api as sm
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC,MultiTaskLassoCV

import scipy
from sklearn.model_selection import KFold

import control_function.control_function as control_function

def data_shift(shift, threshold, mob, mobind, days_considered, causal, type_of_function):
    '''
    mob is used to change the mobility
    'g' is used for google's mobility numbers, 'a' is used for apple's mobility numbers
    mobind is used to give the index of the different dimensions of mobility
    'g' : use dimension [2,3] (more on it in the supplementary), 'a', use [0,2] (more about it in the supplementary)
    '''

    m_types = ['Retail and Recreation', 'Grocery and Pharmacy', 'Parks', 'Transit Stations', 'Workplaces',
               'Residential']
    a_types = ['Driving','Transit','Walking']

    # calling the function to collect data
    mask,mobility,apple,test,trend,disease,growthrate,policy_data,national_data,province_data = data_collection(threshold)

    # this function is to be used when we use control function approach
    # the variable is called causal (when we want to use control function, put causal = 1)
    # the following function is only called in the model if the causal = 1, else it is deleted
    y_mask_pred = control_function(type_of_function, days_considered,mask)

    data = []  # week from 1st 100 case,dailycases,mobility,mask,fear
    count = -1
    for j in range(numb_c):
        gg1 = start[j]
        weeks_considered = int(days_considered / 7)

        # the following if condition is used to avoid the inverse matrix going unstable when days_considered%7 == 0
        if days_considered % 7 == 0:
            weeks_considered = int(days_considered / 7) - 1
        gg2 = min(220, start[j] + days_considered)  # 60 days, w = 6, 90 days, w = 12

        for t in range(gg1, gg2):
            count += 1
            week_numb = min(12, int((t - gg1) / 7))

            temp = [0] * numb_c
            temp[j] = 1

            x = [countries[j], t]

            # the following lines give different transformations to mask numbers
            # the first term is the error numbers from the control function approach
            if type_of_function == 'sqrt':
                x.append(np.sqrt(1 + mask[t - shift, j]) - y_mask_pred.iloc[count])
                x.append(np.sqrt(1 + mask[t - shift, j]))
            elif type_of_function == 'raw':
                x.append(mask[t - shift, j] - y_mask_pred.iloc[count])
                x.append(mask[t - shift, j])
            elif type_of_function == 'log':
                x.append(np.log(1 + mask[t - shift, j]) - y_mask_pred.iloc[count])
                x.append(np.log(1 + mask[t - shift, j]))

            # these numbers are used to select which mobility are considered (google or apple's)
            if mob == 'g':
                for m in mobind:
                    x.append(mobility[j, t - shift, m])
            else:
                for m in mobind:
                    x.append(apple[j, t - shift, m])

            # the following variables add the week of the day fixed effects
            # week 0  = first 7 days from the start[j] of a country j
            week_of_period = [0] * 20
            week_of_period[week_numb] = 1
            for www in range(weeks_considered):
                x.append(week_of_period[www])

            # these are the two y (response variables)
            x.append(growthrate[t, j])
            x.append(dailycases[t, j])

            x.append(test[t - shift, j])
            x.append(trend[t - shift, j])

            for p in range(19):
                x.append(policy_data[j, p, t - shift])

            for c in range(numb_c - 1):
                x.append(temp[c])
            data.append(x)

    cols = ['country', 'day']
    tt = ['Driving', 'Transit', 'Walking']
    cols.append('Mask_residuals')
    cols.append('Mask')
    # '''
    if mob == 'g':
        for m in mobind:
            cols.append('Mobility ' + m_types[m])
    else:
        for m in mobind:
            cols.append('Mobility ' + a_types[m])

    for www in range(weeks_considered):
        cols.append('week' + str(www))
    cols.append('growth')
    cols.append('cases')
    cols.append('Testing')
    cols.append('Trend')

    for p in govtpolicies:
        cols.append(p)
    for c in range(numb_c - 1):
        cols.append(countries[c])
    data = pd.DataFrame(data, columns=cols)

    x = pd.DataFrame.copy(data)
    del x['growth']
    del x['cases']
    if causal != 1:
        del x['Mask_residuals']

    '''
    following lines of codes considers the policies considered and not considred in this work
    We combine of the the government policies
    Some of the policies not relevant to the work as NPI were not considered
    '''
    x['Health Resources'] += x['Health Testing'] + x['Health Monitoring']
    x['Health Resources'] /= np.max(x['Health Resources'])
    del x['Health Testing']
    del x['Health Monitoring']
    del x['Hygiene']

    del x['Other Policy Not Listed Above']
    del x['Anti-Disinformation Measures']
    del x['Declaration of Emergency']
    del x['Public Awareness Measures']
    del x['New Task Force, Bureau or Administrative Configuration']
    x['Restriction and Regulation of Businesses'] += x['Restriction and Regulation of Government Services']
    del x['Restriction and Regulation of Government Services']
    del x['Lockdown']
    del x['Curfew']

    if scenario in [1, 2]:
        del x['lagged']

    x = sm.add_constant(x)
    y = data['growth']

    # collecting data, ensuring that no number is greater than the number of days in the dataset
    x_train = x[x['day'] < 220]
    y_train = y[x['day'] < 220]

    x_train_copy = pd.DataFrame.copy(x_train)
    del x_train['day']
    del x_train['country']

    return (x_train, x_train_copy, y_train)

def data_exponential_smoothing(smoothing_window, threshold, mob, mobind, days_considered, causal, type_of_function):
    '''
    mob is used to change the mobility
    'g' is used for google's mobility numbers, 'a' is used for apple's mobility numbers
    mobind is used to give the index of the different dimensions of mobility
    'g' : use dimension [2,3] (more on it in the supplementary), 'a', use [0,2] (more about it in the supplementary)
    '''

    m_types = ['Retail and Recreation', 'Grocery and Pharmacy', 'Parks', 'Transit Stations', 'Workplaces',
               'Residential']
    a_types = ['Driving','Transit','Walking']

    # calling the function to collect data
    mask,mobility,apple,test,trend,disease,growthrate,policy_data,national_data,province_data = data_collection(threshold)

    # this function is to be used when we use control function approach
    # the variable is called causal (when we want to use control function, put causal = 1)
    # the following function is only called in the model if the causal = 1, else it is deleted
    y_mask_pred = control_function(type_of_function, days_considered,mask)

    data = []
    count = -1
    for j in range(numb_c):
        gg1 = start[j]
        weeks_considered = int(days_considered / 7)

        # the following if condition is used to avoid the inverse matrix going unstable when days_considered%7 == 0
        if days_considered % 7 == 0:
            weeks_considered = int(days_considered / 7) - 1
        gg2 = min(220, start[j] + days_considered)  # 60 days, w = 6, 90 days, w = 12

        for t in range(gg1, gg2):
            count += 1
            week_numb = min(12, int((t - gg1) / 7))

            temp = [0] * numb_c
            temp[j] = 1

            x = [countries[j], t]

            # the following lines give different transformations to mask numbers
            # the first term is the error numbers from the control function approach
            if type_of_function == 'log':
                x.append(-y_mask_pred.iloc[count] + np.log(
                    1 + np.average(mask[t - smoothing_window:t, j], weights=weightage)))
                x.append(np.log(1 + np.average(mask[t - smoothing_window:t, j], weights=weightage)))
            elif type_of_function == 'raw':
                x.append(
                    -y_mask_pred.iloc[count] + np.average(mask[t - smoothing_window:t, j], weights=weightage))
                x.append(np.average(mask[t - smoothing_window:t, j], weights=weightage))
            elif type_of_function == 'sqrt':
                x.append(-y_mask_pred.iloc[count] + np.sqrt(
                    1 + np.average(mask[t - smoothing_window:t, j], weights=weightage)))
                x.append(np.sqrt(1 + np.average(mask[t - smoothing_window:t, j], weights=weightage)))

            if mob == 'g':
                for m in mobind:
                    x.append(np.average(mobility[j, t - smoothing_window:t, m], weights=weightage))
            else:
                for m in mobind:
                    x.append(np.average(apple[j, t - smoothing_window:t, m], weights=weightage))

                # x.append(w)
            week_of_period = [0] * 15
            week_of_period[week_numb] = 1
            for www in range(weeks_considered):
                x.append(week_of_period[www])

            x.append(growthrate[t, j])
            x.append(dailycases[t, j])

            x.append(np.average(test[t - smoothing_window:t, j], weights=weightage))
            x.append(np.average(trend[t - smoothing_window:t, j], weights=weightage))

            for p in range(len(govtpolicies)):
                # x.append(np.average(policy_data[j,p,t-shift-smoothing_window:t-shift],weights=weightage))
                # either select the one above to smooth out policy data or keep the count as shown below
                cc = national[j, p, t - shift + 4] - national[j, p, t - 2 * shift + 4]
                cc += province[j, p, t - shift + 4] - province[j, p, t - 2 * shift + 4]
                x.append(cc * 0.5 / (1 + shift))
            # '''
            for c in range(numb_c - 1):
                x.append(temp[c])
            data.append(x)

    cols = ['country', 'day']
    tt = ['Driving', 'Transit', 'Walking']
    cols.append('Mask_residuals')
    cols.append('Mask')
    # '''
    if mob == 'g':
        for m in mobind:
            cols.append('Mobility ' + m_types[m])
    else:
        for m in mobind:
            cols.append('Mobility ' + a_types[m])

    for www in range(weeks_considered):
        cols.append('week' + str(www))
    cols.append('growth')
    cols.append('cases')
    cols.append('Testing')
    cols.append('Trend')

    for p in govtpolicies:
        cols.append(p)
    for c in range(numb_c - 1):
        cols.append(countries[c])
    data = pd.DataFrame(data, columns=cols)

    x = pd.DataFrame.copy(data)
    del x['growth']
    del x['cases']
    if causal != 1:
        del x['Mask_residuals']

    '''
    following lines of codes considers the policies considered and not considred in this work
    We combine of the the government policies
    Some of the policies not relevant to the work as NPI were not considered
    '''
    x['Health Resources'] += x['Health Testing'] + x['Health Monitoring']
    x['Health Resources'] /= np.max(x['Health Resources'])
    del x['Health Testing']
    del x['Health Monitoring']
    del x['Hygiene']

    del x['Other Policy Not Listed Above']
    del x['Anti-Disinformation Measures']
    del x['Declaration of Emergency']
    del x['Public Awareness Measures']
    del x['New Task Force, Bureau or Administrative Configuration']
    x['Restriction and Regulation of Businesses'] += x['Restriction and Regulation of Government Services']
    del x['Restriction and Regulation of Government Services']
    del x['Lockdown']
    del x['Curfew']

    if scenario in [1, 2]:
        del x['lagged']

    x = sm.add_constant(x)
    y = data['growth']

    # collecting data, ensuring that no number is greater than the number of days in the dataset
    x_train = x[x['day'] < 220]
    y_train = y[x['day'] < 220]

    x_train_copy = pd.DataFrame.copy(x_train)
    del x_train['day']
    del x_train['country']

    return (x_train, x_train_copy, y_train)

def linear_regression():
    mob = 'g'
    mobind = [2, 3]
    days_considered = 60
    causal = 0
    type_of_function = 'log'

    x_train, x_train_copy, y_train = data_shift(shift, threshold,mob, mobind, days_considered, causal,
                                                type_of_function,mask,mobility,trend,test,
                                                growthrate,policy_data)

    model = sm.OLS(y_train, x_train)
    res = model.fit()

    # to get the combined effect of masks+mobility+NPI
    EFFECTS = combined_effect(res,type_of_function,x_train)
    return(res,EFFECTS)

def lasso_regression():
    mob = 'g'
    mobind = [0,1,2,3,4,5]
    days_considered = 60
    causal = 0
    type_of_function = 'log'

    x_train, x_train_copy, y_train = data_shift(shift, threshold,mob, mobind, days_considered, causal,
                                                type_of_function,mask,mobility,trend,test,
                                                growthrate,policy_data)

    # using 5-fold cross validation to select the best parameters for lasso
    model_aic = LassoCV(cv=5)
    model_aic.fit(x_train, y_train)
    alpha_aic_ = model_aic.alpha_

    model = Lasso(alpha=alpha_aic_).fit(x_train, y_train)
    coeff = pd.DataFrame(model.coef_, columns=['Lasso'])
    EFFECTS = combined_effect(res,type_of_function,x_train)

    return(coeff,EFFECTS)

# CI function is used for a given array
def CI(sample):
    # select what bounds is required (95% confidence interval or 25th and 75th percentile)
    confidence_level = 0.95
    degrees_freedom = sample.size - 1
    sample_mean = np.mean(sample)
    sample_standard_error = np.std(sample)

    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    a = [0]*3
    a[0],a[1],a[2] = confidence_interval[0],sample_mean,confidence_interval[1]

    # when computing active cases (forward looking model, confidence interval have to be computed by using percentiles)
    # a[0],a[1],a[2] = np.percentile(sample,25),np.percentile(sample,50),np.percentile(sample,75)

    return(confidence_interval)

def combined_effect(res,type_of_function,x_train):

    # calculating covariance and correlation matrix for the the beta coefficients
    # using residual to estimate variance (sse)

    ssquare = np.mean((y_train - res.predict(x_train)) ** 2)
    coefficients = np.array(res.params).reshape(-1, 1)

    if 'Mask_residuals' in list(x_train.columns):
        residual_index = list(x_train.columns).index('Mask_residuals')
        coefficients = np.delete(coefficients, residual_index, 0)
        del x_train['Mask_residuals']

    cov = ssquare * np.linalg.inv(np.matmul(x_train.values.T, x_train.values))

    cholesky = np.linalg.cholesky(cov)
    cholesky = np.array(cholesky)

    mask_ind = list(x_train.columns).index('Mask')
    total_ind = [mask_ind]
    mobility_ind = []        # index for mobility
    for c, loc in enumerate(list(x_train.columns)):
        if 'Mobility' in loc:
            mobility_ind.append(c)
            total_ind.append(c)

    npi_ind = []             # index for NPI
    for c, pol in enumerate(list(x_train.columns)):
        if pol in govtpolicies:
            npi_ind.append(c)
            total_ind.append(c)

    # adding perturbations to the cholesky matrix
    epsilon = 0.001
    d = len(x_train.columns)
    cholesky = cholesky + epsilon * np.identity(d)

    # generating 10000 samples
    n = 10000
    u = np.random.normal(loc=0, scale=1, size=d * n).reshape(d, n)
    beta_samples = coefficients + np.matmul(cholesky, u)

    # converting mobility to negative and masks to exponential/sqrt form (interpretation of coefficients for mask: supplementary)
    beta_samples[mobility_ind, :] = np.negative(beta_samples[mobility_ind, :])
    if type_of_function == 'log':
        beta_samples[mask_ind, :] = -(1 - np.power(2, beta_samples[mask_ind, :]))
    elif type_of_function == 'sqrt':
        beta_samples[mask_ind, :] = (np.sqrt(2) - 1) * beta_samples[mask_ind, :]

    mask_effect = [0] * 3
    mobility_effect = [0] * 3
    npi_effect = [0] * 3
    total_effect = [0] * 3

    mask_samples = beta_samples[mask_ind, :]
    mobility_samples = np.sum(beta_samples[mobility_ind], axis=0)
    npi_samples = np.sum(beta_samples[npi_ind], axis=0)
    total_samples = np.sum(beta_samples[total_ind], axis=0)

    mask_effect[0], mask_effect[2] = CI(mask_samples)
    mobility_effect[0], mobility_effect[2] = CI(mobility_samples)
    npi_effect[0], npi_effect[2] = CI(npi_samples)
    total_effect[0], total_effect[2] = CI(total_samples)

    mask_effect[1] = np.mean(mask_samples)
    mobility_effect[1] = np.mean(mobility_samples)
    npi_effect[1] = np.mean(npi_samples)
    total_effect[1] = np.mean(total_samples)

    return (total_effect, mask_effect, mobility_effect, npi_effect)
