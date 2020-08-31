import time
from datetime import datetime
import numpy as np
import pandas as pd
import glob, os
from sklearn import metrics

import statsmodels.api as sm
from sklearn import linear_model

def control_function(type_of_function, days_considered,disease,mask):

    dis = pd.read_csv('.../data files//disease.csv')
    disease = np.zeros((6, numb_c))
    # index 0 - 3 includes deaths, index 3-5 contains total confirmed cases
    # converted to per thousand people
    for j in range(numb_c):
        disease[0, j] = dis.iloc[j, 1] * 1000 / population[j, 0]
        disease[1, j] = dis.iloc[j, 2] * 1000 / population[j, 0]
        disease[2, j] = dis.iloc[j, 3] * 1000 / population[j, 0]
        disease[3, j] = dis.iloc[j, 4] * 1000 / population[j, 0]
        disease[4, j] = dis.iloc[j, 5] * 1000 / population[j, 0]
        disease[5, j] = dis.iloc[j, 6] * 1000 / population[j, 0]

    m1 = np.median(disease[:, 0])
    m2 = np.median(disease[:, 1])
    m3 = np.median(disease[:, 2])

    # dmodel is the dataframe with the data for modeling masks
    dmodel = []
    for j in range(numb_c):
        for t in range(82, 220):
            if type_of_function == 'sqrt':
                xx = [np.log(1 + int((t - 82) / 7)), np.sqrt(1 + mask[t, j]), int(disease[0, j] > m1),
                      int(disease[1, j] > m2), int(disease[2, j] > m3)]
            elif type_of_function == 'log':
                xx = [np.log(1 + int((t - 82) / 7)), np.log(1 + mask[t, j]), int(disease[0, j] > m2),
                      int(disease[1, j] > m2), int(disease[2, j] > m3)]
            elif type_of_function == 'raw':
                xx = [np.log(1 + int((t - 82) / 7)), mask[t, j], int(disease[0, j] > m1), int(disease[1, j] > m2),
                      int(disease[2, j] > m3)]
            dmodel.append(xx)

    cols = ['day', 'mask', 'Sars', 'H1N1', 'Mers']

    dmodel = pd.DataFrame(dmodel, columns=cols)
    yy = dmodel['mask']
    del dmodel['mask']
    del dmodel['day']
    dmodel['const'] = 1

    # disease model is the OLS regression model
    disease_model = sm.OLS(yy, dmodel)
    disease_model = disease_model.fit()

    dmodel = []
    for j in range(numb_c):
        gg1 = start[j]
        gg2 = min(220, start[j] + days_considered)
        for t in range(gg1, gg2):
            if type_of_function == 'sqrt':
                xx = [np.log(1 + int((t - 82) / 7)), np.sqrt(1 + mask[t, j]), int(disease[0, j] > m1),
                      int(disease[1, j] > m2), int(disease[2, j] > m3)]
            elif type_of_function == 'log':
                xx = [np.log(1 + int((t - 82) / 7)), np.log(1 + mask[t, j]), int(disease[0, j] > m1),
                      int(disease[1, j] > m2), int(disease[2, j] > m3)]
            elif type_of_function == 'raw':
                xx = [np.log(1 + int((t - 82) / 7)), mask[t, j], int(disease[0, j] > m1), int(disease[1, j] > m2),
                      int(disease[2, j] > m3)]

            dmodel.append(xx)

    cols = ['day', 'mask', 'Sars', 'H1N1', 'Mers']

    dmodel = pd.DataFrame(dmodel, columns=cols)
    del dmodel['mask']
    del dmodel['day']
    dmodel['const'] = 1
    y_mask_pred = disease_model.predict(dmodel)

    return (y_mask_pred)