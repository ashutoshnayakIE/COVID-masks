import search_function
import regression_models
import average_country

def main():
    '''
    ------------------- VARIABLE DEFINITION ------------------------
    mask : 225 days x 24 countries (survey data on  mask wearing)
    mobility: 24 countries x 225 days x 6 (mobility in 6 different locations from google community mobility reports)
    apple: 24 countries x 225 days x 2 (mobility in 3 different mobility types from apple community mobility reports)
    policy_data: 24 countries x 19 policies x 225 days (0/1 data for government policies normalized for population)
    national_data: 24 countries x 19 policies x 225 days (0/1 data for government policies at national level)
    province_data: 24 countries x 19 policies x 225 days (0/1 data for government policies at provincial level)
    test: 225 days x 24 countries (number of tests per thousand people)
    trend: 225 days x 24 countries (google trends for the keyword "coronavirus')

    country fixed effects: fixed effects for countries
    week fixed effects: fixed effect for week of the day

    shift: lag considered in this research
    causal: 1 if using control function approach, 0 otherwise
    window: window of days considered for exponential smoothing
    days_considered: 60 in our analysis. However, the model performance can be checked by changing this variable
    mob: mobility we use for the model ('g' = google and 'a' = apple)
    mobind: [2,3] for google and [0,2] for apple as explained in Section social mobility in supplementary
    type_of_function: is the selection of transformation for masks (log, raw (linear) or sqrt function)

    growthrate: output variables: 225 days x 24 countries (shows the growth rate for different countries)
    dailycases: output variables: 225 days x 24 countries (shows the daily active cases for different countries)

    --------------------- CODING STRUCTURE---------------------------
    Run the model sequentially as shown in the steps below, you can change the parameters to see diff in parameter values
    All the functions should run independently, so comment other functions if you want to check one particular function
    '''

    # select the threshold value
    # this function prints the log likelihood values for threshold over different shifts (we use 0.2 in our analysis)
    search_function.finding_th()

    # after finding the threshold we want to use for the models (we selected 0.2), we can select the best shift value
    # the following function prints the average MAPE for 10-fold cross validation
    # we select the shift with the lowest number for MAPE : 9 days
    search_function.finding_shift()

    # after selecting th and shift, we can run the linear regression model to estimate the parameter coefficients
    # to print the results from the linear regression model, uncomment the line: print(res.summary())
    # it has further two subsections: confidence interval and combined effects. Confidence interval function finds the
    # upper and lower confidence interval around the mean of the prediction. combined effects uses Krinsky-Robb simulation
    # method to calculate the combined effect of masks, mobility, NPI and (mask+mobiliyt+NPI)
    res,EFFECTS = regression_models.linear_regression()

    # prints the result summary for the model
    print(res.summary())
    # print the combined effects of each of the measures (mask, mobility, NPI, mask+mobiliyt+NPI)
    print(EFFECTS)


    '''
    ------------------- ROBUSTNESS CHECKS ---------------------------
    We perform the following robustness checks in our analysis. Most of these robustness checks can be done by changing
    the model parameters (as discussed in the variable definition above) 
    
    1. changing the th value
    2. changing the shift values
    3. changing the mobility (with 'g' use [2,3], with 'a' use [0,2])
    4. control function approach: change causal from 0 to 1
    5. days_considered: change the days (in the supplement, we present results for days 35, 45 ,..., 85)
    6. type_of_function (transforming the mask numbers as log, linear, or sqrt)
    7. lasso regression model to select the mobility dimensions
    '''

    '''
    ------------------- ANALYZING RESULTS ----------------------------
    We present the basic codes for all the analysis performed in our analysis
    Anyone, interested in replicating the results need to write short codes on changing different parameter values
    for example, if you want to test the combined effect of masks under different transformations of mask numbers,
    Similarly, for all the following analysis, please change the numbers for variables of interest to observe different results
    '''
    # analysing the results on the effect of masks by using an average country
    # results provide actcases: active cases on different days for a hypothetical country
    actcases = average_country.average_country_results()
    # print(actcases)

    '''
    ------------------ SUB FUNCTIONS USED ---------------------------
    CI() is used to calculate confidence intervals for a given array (it can be 95% confidence interval or percentiles)
    combined_effect() is used to calculate the combined effect of masks, mobility or NPI
    average_bounds() is used to used to calculate the bounds for active cases in average country: Krinsky-Robb method
    '''


if __name__ == "__main__":
    main()