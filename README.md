# COVID-19 and masks
A repository for data and codes in the study to estimate the effect of masks using reduced form statistical methods

# Data
All the data files are provided in the .npy format. Data has been obtained from the sources below. Dimensions of each data are noted next to the data. We collect data from December 1, 2019 for all the data set for which data was available. Data from Dec 1 2019 - July 12 2020 has 225 days. We mainly focus on data from Feb 21, 2020 and July 8, 2020 as all the data were available for this time period (which is also the period of the analysis in this work)

1) Mask data (dimension: 225 days x 24 countries):<br>
https://today.yougov.com/topics/international/articles-reports/2020/03/17/personal-measures-taken-avoid-covid-19

2) Mobility data:<br>
Google: https://www.google.com/covid19/mobility/ (dimension: 24 countries x 225 days x 6 locations)
Apple : https://covid19.apple.com/mobility       (dimension: 24 countries x 225 days x 3 types)

3) Government policies (dimension: 24 countries x 19 policies x 225 days):<br>
https://www.coronanet-project.org/

4) Trend data (dimension: 225 days x 24 countries):<br>
https://trends.google.com/trends/explore?q=coronavirus&geo=US

5) Testing data (dimension: 225 days x 24 countries):<br>
https://ourworldindata.org/coronavirus-testing

6) Daily confirmed/recovered cases (dimension: 225 days x 24 countries):<br>
https://github.com/CSSEGISandData/COVID-19

# Code
All the codes are included here for running the model. Run the following code:
