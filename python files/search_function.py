import time
from datetime import datetime
import numpy as np
import pandas as pd
import glob, os
from sklearn import metrics

import statsmodels.api as sm
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC,MultiTaskLassoCV
from sklearn.model_selection import KFold

def finding_th(shift, mob, mobind, days_considered, causal, type_of_function,mask,mobility,trend,test,growthrate,policy_data):
    vals = [0]*30

    # we run the model for different shifts (shift = 0 days to 14 days)
    # so as to observe that a particular th does not perform by chance
    # even though the best value for th is observed at th=0.28, we select th=0.2 (Supplementary)
    for s in range(15):
        for th in range(30):
            x_train,x_train_copy,y_train,x_test,x_test_copy,y_test = data_shift(shift, th/100, mob, mobind, days_considered, causal, type_of_function,mask,mobility,trend,test,growthrate,policy_data)
            model = sm.OLS(y_train,x_train)
            res   = model.fit()
            vals[p]  += res.llf
    print(vals.index(max(vals)))

def finding_shift():

    # checking the shift values from 0 days to 14 days
    # the best value for shift is obtained at shift = 9 days (minimum MAPE)
    for s in range(15):
        x_train, x_train_copy, y_train, x_test, x_test_copy, y_test = data_shift(shift, th, mob, mobind, days_considered, causal, type_of_function,mask,mobility,trend,test,growthrate,policy_data)

        # using KFold Cross validation and using the average Mean Absolute Percetage Error (MAPE) to identify the best shift
        kf = KFold(n_splits=10)
        kf.get_n_splits(x_train)
        v = 0
        for train_index, test_index in kf.split(x_train):
            X_t1, X_t2 = x_train.iloc[train_index], x_train.iloc[test_index]
            Y_t1, Y_t2 = y_train.iloc[train_index], y_train.iloc[test_index]

            model = sm.OLS(Y_t1, X_t1)
            res = model.fit()
            Y_pred = res.predict(X_t2)
            v += np.mean(np.abs((Y_t2 - Y_pred) / Y_t2))
        print(s, v)
