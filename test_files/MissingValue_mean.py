import pandas as pd
import numpy as np


# ساخت دیتای نمونه
np.random.seed(0)
df = pd.DataFrame({
    "x": [12.1, 14, np.nan, 10, 2, 23, np.nan, 15, 7, 11],
    "y": [0.2, 0.41, 0.32, 0.19, 0.05, 0.67, 0.11, 0.32, np.nan, 0.23],
})

#print("data: ")
#print(df)

#print(np.sum(df.isnull(), axis = 0))
#print(df.shape)
#print(df.info())
# mvs_summary = pd.DataFrame({'freq' : np.sum(df.isnull(), axis = 0)})
# mvs_summary['pct'] = round(mvs_summary['freq']/df.shape[0] * 100, 1)
# mvs_summary.sort_values(by='pct', ascending=False)
# #print(mvs_summary)

# df.loc[:, 'mvs'] = np.sum(df.isnull(), axis = 1)
# print(df.sort_values(by = 'mvs', ascending = False))

data_mean = df.fillna(df.mean())
print(data_mean)
data_corr = round(data_mean.corr(),2)
print(data_corr)
