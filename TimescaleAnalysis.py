'''
Description: 
    Analyze different timescales..

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    May. 7 2020
'''

import statsmodels.api as sm
import pandas
import patsy
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df = df.dropna()
print(df[-5:])
y, X = patsy.dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())
print(sm.stats.linear_rainbow(res))

sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],data=df, obs_labels=False)
plt.show()