import time
from datetime import datetime
import numpy as np
import pandas as pd
import glob, os

def data_collection(th):

    # th is the threshold for which we start collecting data (th = % of peak cases in a country)

    # countries considered in this research
    countries = ['Australia', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy',
                 'Japan', 'Malaysia', 'Mexico', 'Norway', 'Philippines', 'UAE', 'Saudi Arabia', 'Singapore', 'Spain',
                 'Sweden', 'Taiwan', 'Thailand', 'United Kingdom', 'USA', 'Vietnam']

    numb_c = len(countries)
    mask  = np.load(".../data files//mask.npy")
    mobility = np.load(".../data files//mobility.npy")
    apple     = (np.load(".../data files//apple.npy") - 100)    # normalizing the numbers of apple from 0 - 1

    m_types = ['Retail and Recreation', 'Grocery and Pharmacy', 'Parks', 'Transit Stations', 'Workplaces',
               'Residential']

    dailyConfirmedCases = np.load(".../data files//dailyConfirmedCases.npy")
    recoveredCases      = np.load(".../data files//recoveredCases.npy")
    test  = np.load(".../data files//test.npy")
    trend = np.load(".../data files//trend.npy")

    # contains information on population,density and urban population of each country
    # we only use the information on total population
    population = np.load(".../data files//population.npy")
    population[:, 3] /= 100

    # using 7 day average for cumulative daily confirmed cases and recovery cases
    # using this to find out the daily active infectious cases
    for j in range(numb_c):
        for t in range(6, 220):
            smoothcases[t, j] = np.mean(smoothcases[t - 3:t + 4, j])
            smoothcases[t, j] = max(smoothcases[t - 1, j], smoothcases[t, j])
            smoothrecover[t, j] = np.mean(smoothrecover[t - 3:t + 4, j])
            smoothrecover[t, j] = max(smoothrecover[t - 1, j], smoothrecover[t, j])
            activecases[t, j] = max(0, smoothcases[t, j] - smoothrecover[t, j])
            growthrate[t, j] = np.log(1 + activecases[t, j]) - np.log(1 + activecases[t - 1, j])
            dailycases[t, j] = smoothcases[t, j] - smoothcases[t - 1, j]

    # finding the start time for each country for data collection
    # it is based on the selection of th
    for j in range(numb_c):
        for t in range(70, 220):
            if np.mean(dailycases[t:t + 7, j]) >= threshold * np.max(dailycases[0:220, j]) and start[j] == 0:
                start[j] = t
                break
        start[j] = max(82, start[j])

    policy_data, national_data, province_data = policy_data()
    return(mask,mobility,apple,test,trend,growthrate,policy_data,national_data,province_data)



def policy_data():
    # collecting data from government policies
    # use the file : policies.csv
    # before collecting the data on policy, first we collect information on population of each states (in the data set)

    provincial_data = {}
    population_data = pd.read_csv(".../data files//population_data.csv", encoding='utf-8')

    for j in range(numb_c):
        provincial_data[countries[j]] = {}
        provincial_data[countries[j]]['population'] = int(population[j, 0])

    for s in range(len(population_data)):
        country = population_data['country'].iloc[s]
        if 'America' in country:
            country = 'USA'
        elif 'Emirates' in country:
            country = 'UAE'
        state = population_data['state'].iloc[s]
        pop = int(population_data['population'].iloc[s])
        provincial_data[country][state] = pop

    policy = pd.read_csv(".../data files//policies.csv")

    policy_data   = np.zeros((numb_c, len(govtpolicies), 225))   # includes population normalized data
    national_data = np.zeros((numb_c, len(govtpolicies), 225))   # information of policies at national level
    province_data = np.zeros((numb_c, len(govtpolicies), 225))   # information of policies at provincial level

    for s in range(len(policy)):
        country = policy['country'].iloc[s]
        if 'America' in policy['country'].iloc[s]:
            country = 'USA'
        elif 'Emirates' in policy['country'].iloc[s]:
            country = 'UAE'
        c = countries.index(country)
        p = policy['type'].iloc[s]

        sd = int(policy['start_date'].iloc[s])
        ed = int(max(sd, min(225, policy['end_date'].iloc[s])))
        indp = 1000

        # bins have been used to assign the value between 0 - 1 based on the intensity of the policy
        # in this work, we only consider if they were implemented (1) or not (0)
        bins = policy['binary'].iloc[s]
        if bins > 0:
            bins = 1
        else:
            bins = 0

        # cumulatively adding the count of the policies
        if p in ['Health Resources', 'Health Testing', 'Health Monitoring', 'Hygiene', 'Anti-Disinformation Measures',
                 'Other Policy Not Listed Above', 'New Task Force, Bureau or Administrative Configuration',
                 'Public Awareness Measures']:
            policy_data[c, govtpolicies.index(p), sd:] += 1
            if policy['init_country_level'].iloc[s] == 'National':
                national_data[c, govtpolicies.index(p), sd:] += 1
            if policy['init_country_level'].iloc[s] == 'Provincial':
                province_data[c, govtpolicies.index(p), sd:] += 1

        # have clear 0/1 for the policy, weighing them in terms of the population of that country
        if p in ['Internal Border Restrictions', 'External Border Restrictions', 'Lockdown', 'Curfew',
                 'Closure and Regulation of Schools', 'Restriction and Regulation of Businesses',
                 'Declaration of Emergency', 'Restrictions of Mass Gatherings',
                 'Restriction and Regulation of Government Services']:

            if policy['init_country_level'].iloc[s] == 'National':
                policy_data[c, govtpolicies.index(p), sd:ed] += bins
                national_data[c, govtpolicies.index(p), sd:ed] += bins
            if policy['init_country_level'].iloc[s] == 'Provincial':
                if policy['province'].iloc[s] in provincial_data[country].keys():
                    state = policy['province'].iloc[s]
                    r = provincial_data[country][state] * bins / (provincial_data[country]['population'])
                    policy_data[c, govtpolicies.index(p), sd:ed] += r
                    province_data[c, govtpolicies.index(p), sd:ed] += r

        # quarantine
        if p in ['Quarantine']:
            sc = policy['type_sub_cat'].iloc[s]  # sub-class
            indp = govtpolicies.index('Quarantine')
            if indp < 1000:
                if policy['init_country_level'].iloc[s] == 'National':
                    policy_data[c, indp, sd:ed] += bins
                    national_data[c, indp, sd:ed] += bins
                if policy['init_country_level'].iloc[s] == 'Provincial' and policy['province'].iloc[s] in list(
                        population_data['state']):
                    state = policy['province'].iloc[s]
                    r = provincial_data[country][state] * bins / (provincial_data[country]['population'])
                    policy_data[c, indp, sd:ed] += r
                    province_data[c, indp, sd:ed] += r

        # social distancing
        if p in ['Social Distancing']:
            sc = policy['type_sub_cat'].iloc[s]
            indp = govtpolicies.index('Social Distancing')
            '''
            if 'All public spaces' in sc:
                indp = govtpolicies.index('Social Distancing-all Public area')
            elif '-' in sc:
                indp = govtpolicies.index('Social Distancing-other')
            '''
            if indp < 1000:
                if policy['init_country_level'].iloc[s] == 'National':
                    policy_data[c, indp, sd:ed] += bins
                    national_data[c, indp, sd:ed] += bins
                if policy['init_country_level'].iloc[s] == 'Provincial' and policy['province'].iloc[s] in list(
                        population_data['state']):
                    state = policy['province'].iloc[s]
                    r = provincial_data[country][state] * bins / (provincial_data[country]['population'])
                    policy_data[c, indp, sd:ed] += r
                    province_data[c, indp, sd:ed] += r

    # we normalize the data for every country
    for j in range(numb_c):
        for k in range(len(govtpolicies)):
            if govtpolicies[k] in ['Hygiene']:
                policy_data[j, k, :] /= (0.0001 + np.max(policy_data[j, k, :]))
            else:
                policy_data[j, k, :] /= (0.0001 + np.max(policy_data[j, k, :]))
            national_data[j, k, :] /= (0.0001 + np.max(national_data[j, k, :]))
            province_data[j, k, :] /= (0.0001 + np.max(province_data[j, k, :]))

    return (policy_data,national_data,province_data)
