#To install matplotlib
#%pip install matplotlib

#Import matplotlib
import matplotlib
#Get the version of matplotlib
print(matplotlib.__version__)

#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Example: Plot f(x) = x ^ 2 * e ^ (5 - x), 0 <= x <= 5
def f(x):
    return x ** 2 * np.exp(5 - x)

#Generate x and y for the curve
x = np.linspace(0, 5, 100)
y = f(x)

#Plot the curve
plt.plot(x, y)
plt.show()

#Add title, axis label, add grid and change color
plt.plot(x, y, color = 'red')
plt.title('f(x) Curve')
plt.xlabel('X')
plt.ylabel('f(x)')
plt.grid()
plt.show()

#Import mtcars dataset
data = pd.read_csv('mtcars.csv')

#Dataframe overview
data.info()

#hEAD
data.head()

#Scatter plot
plt.scatter(x = data['wt'], y = data['mpg'], color = 'red', alpha = 0.7)
plt.title('Miles per Gallon vs. Weight')
plt.xlabel('Weight', fontsize = 12)
plt.ylabel('Miles per Gallon', fontsize = 12)
plt.grid()
plt.show()

#Scatter plot w/ third categorical variable
#Color
cdict = {3: 'red', 4: 'blue', 5: 'green'}
#Marker
mdict = {3: '*', 4: '^', 5: 'o'}
#Scatter plot
for i in data['gear'].unique():
    df = data.loc[data['gear'] == i, ]
    plt.scatter(x = df['wt'], y = df['mpg'], 
                alpha = 0.5, color = cdict[i], marker = mdict[i], label = i)
plt.legend(title = '# of Gears')
plt.title('Miles per Gallon vs. Weight', fontsize = 15)
plt.xlabel('Weight', fontsize = 12)
plt.ylabel('Miles per Gallon', fontsize = 12)
plt.grid()
plt.show()

#Add annotation to plot
plt.scatter(x = data['wt'], y = data['mpg'], color = 'red', alpha = 0.7)
plt.text(4.5, 20,'Sample Text', color = 'blue')
plt.title('Miles per Gallon vs. Weight')
plt.xlabel('Weight', fontsize = 12)
plt.ylabel('Miles per Gallon', fontsize = 12)
plt.grid()
plt.show()

#Add regression line to scatter plot: mpg vs. wt
params = np.polyfit(data['wt'], data['mpg'], 1)
params #mpg = -5.344 * wt + 37.285

#Generate xp and yp for regression line
xp = np.linspace(data['wt'].min(), data['wt'].max(), 100)
yp = np.polyval(params, xp)

#Scatter plot: mpg vs. wt
plt.scatter(x = data['wt'], y = data['mpg'], 
            color = 'red', alpha = 0.7, label = 'data')
plt.title('Miles per Gallon vs. Weight')
plt.xlabel('Weight', fontsize = 12)
plt.ylabel('Miles per Gallon', fontsize = 12)
plt.grid()
#Add regression line
plt.plot(xp, yp, 'black', alpha = 0.8, linewidth = 2, label = 'reg. line')
plt.legend()
plt.show()

#LOWESS (Locally Weighted Scatterplot Smoothing)
#https://en.wikipedia.org/wiki/Local_regression

#To install statsmodels
#%pip install statsmodels

#Implement Lowess algorithm
import statsmodels.api as sm
lowess_res = sm.nonparametric.lowess(data['mpg'],  data['wt'], frac = 2 / 3)
lowess_res

#Scatter plot: mpg vs. wt
plt.scatter(x = data['wt'], y = data['mpg'], 
            color = 'red', alpha = 0.7, label = 'data')
plt.title('Miles per Gallon vs. Weight')
plt.xlabel('Weight', fontsize = 12)
plt.ylabel('Miles per Gallon', fontsize = 12)
plt.grid()

#Add LOWESS line
plt.plot(lowess_res[:, 0], lowess_res[:, 1], 'black', 
         alpha = 0.8, linewidth = 2, label = 'LOWESS')
plt.legend()
plt.show()

#Bar chart: Frequency table for # of gears
freq = data['gear'].value_counts()
print(type(freq))
freq

#Bar chart based on frequency
plt.bar(freq.index, freq)
plt.xticks(freq.index)
plt.title('Dist. of Vehicles \n based on # of Gears')
plt.xlabel('# of Gears')
plt.ylabel('Frequency')
plt.show()

#Histogram
plt.hist(data['mpg'], bins = 10, color = 'red', alpha = 0.7)
plt.title('Histogram of Miles per Gallon')
plt.xlabel('Miles per Gallon')
plt.ylabel('Frequency')
plt.show()

#Box Plot
plt.boxplot(data['mpg'])
plt.title('Boxplot of Miles per Gallon')
plt.xticks([]) #remove the xticks 
plt.ylabel('Miles per Gallon')
plt.show()

#Boxplot for multiple groups
plt.boxplot([data.loc[data['cyl'] == 4, 'mpg'], 
             data.loc[data['cyl'] == 6, 'mpg'],
             data.loc[data['cyl'] == 8, 'mpg']])
plt.title('Boxplot of \n Miles per Gallon vs. # of Cylinders')
plt.xticks(ticks = [1, 2, 3], labels = [4, 6, 8])
plt.xlabel('Number of Cylinders')
plt.ylabel('Miles per Gallon')
plt.show()

data.info()

#Sub-plots
var_ind = [3, 4, 6, 7]
plt.figure(figsize = (10, 8))
plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.scatter(x = data.iloc[: , var_ind[i - 1]], y = data['mpg'])
    plt.title('mpg vs. ' + data.columns[var_ind[i - 1]])
