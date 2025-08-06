#Install pandas
#%pip install pandas
#Import required libraries
import numpy as np
import pandas as pd
#Get the version of pandas
print(pd.__version__)
#Create a series
#Demand for four different products
s1 = pd.Series([1750, 1200, 450, 2000])
s1

#Type
type(s1)

#Series name
s1.name #without name

#Assign new name
s1.name = 'product_demand'
s1

#Corresponding array of a series
s1.values

type(s1.values)

#Indexing
s1.index

#Extract an item from series with index
s1[0]

s1.index = ['p1', 'p2', 'p3', 'p4']
s1.index

#Modifying series
s1['p1'] = 1000
s1

#Add new value
s1['p5'] = 1250
s1

#Boolian operations
s1 > 1000

#Extract elements of a series with boolian operation
s1[s1 > 1000]

#Mathematical operation
s1 * 1.10

s1 = s1 * 1.10
s1

#Find the mean of series
s1.mean()

#Find the standard deviation of series
s1.std()

#  &: and
#  |: or
#  ~: not

s1[(s1 < s1.mean()) & (s1 > 1000)]

s1[s1 <= 1000] = 0
s1

#Create a dataframe
df = pd.DataFrame({'name'  : ['p1', 'p2', 'p3', 'p4'],
                   'demand': [1750, 1200, 450, 2000],
                   'brand' : ['x', 'x', 'y', 'z'],
                   'weight': [150, 200, 1500, 200]})
df

#Type
type(df)

#Shape
df.shape

#Columns
df.columns

#Index
df.index

df.index = ['p1', 'p2', 'p3', 'p4']
df

df.info()

#Extract a column
df.demand

#Extract a column
df['demand']

#Extract several columns
df[['name', 'demand']]

#.iloc[ ] allows us to retrieve rows and columns by position.
df.iloc[0, 1]

#Extract information of first product
df.iloc[0, :]

#Extract demand column
df.iloc[:, 1]

#Extract name & brand columns for p1, p3, and p4
df.iloc[[0, 2, 3], [0, 2]]

#.loc[] selects data by the label of the rows and columns. 
#Extract information of first product
df.loc['p1', :]

#Extract demand column
df.loc[:, 'demand']

#Extract name & brand columns for p1, p3, and p4
df.loc[['p1', 'p3', 'p4'], ['name', 'brand']]

#Check if demand below 1500
df['demand'] < 1500

#Identify products with demand below 1500
df.loc[df['demand'] < 1500, 'name']

#Add new column 
df['price'] = [20, 15, 50, 10]
df

#Add new row
df.loc['p5', :] = ['p5', 1000, 'x', 500, 60]
df

#Change demand of 'p3' into 700
df.loc['p3', 'demand'] = 700
df

#Calculate total revenue for each product
df['revenue'] = df['demand'] * df['price']
df

#Modifying dataframe using conditional selection
#Question: increase price of products below 20 by 5% and
# recalculate monthly revenue
df.loc[df['price'] < 20, 'price'] = 1.05 * df.loc[df['price'] < 20, 'price']
df['revenue'] = df['demand'] * df['price']
df

#Get work directory
import os
os.getcwd()

#Read from work directory
data = pd.read_csv('sample_data.csv')

#Read from desktop
data = pd.read_csv('C:\\Users\\FarzadM\\Desktop\\sample_data.csv')

#Type
type(data)

#Head
data.head()
#Tail
data.tail()
#Shape
data.shape
data.info()


#Missing values?
data.isnull()

#Missing values?
np.sum(data.isnull(), axis = 0)

#Q1: Extract information of all female customers.
data.loc[data['sex'] == 'F', :]

#Q2: Extract  'sex', 'age', and 'income' columns
#    of all female customers.
data.loc[data['sex'] == 'F', ['sex', 'income', 'age']]

#Q3: Extract information of all female customers who are bellow 40 years old.
data.loc[(data['sex'] == 'F') & (data['age'] < 40), :]

#Q4: Extract  'sex', 'age', and 'income' columns
#    for male customers who are above 50 years old
data.loc[(data['age'] > 50) & (data['sex'] == 'M'), 
         ['sex', 'age', 'income']]

#Q5: What percentage of customers are female?
round(np.sum(data['sex'] == 'F') / data.shape[0] * 100, 2)

#Q6: Calculate percentage of missing values in is_employed.
np.sum(data['is_employed'].isnull()) / data.shape[0] * 100

#Q7: What percentage of customers are employed?
round(sum(data['is_employed'] == True) / sum(data['is_employed'].notnull()) * 100, 2)

#Q8: What percentage of customers are between 25 and 35?
round(np.sum((data['age'] > 25) & 
             (data['age'] < 35)) / data.shape[0] * 100, 2)

#Q9: What percentage of male customers are above 30?
round(np.sum((data['sex'] == 'M') & 
             (data['age'] > 30)) / np.sum(data['sex'] == 'M') * 100, 2)

#Q10: Extract those customers from Florida 
#     who are older than 75% of all customers.
data.loc[(data['state_of_res'] == 'Florida') & 
         (data['age'] > np.quantile(data['age'], 0.75)), :]

#3rd quantile of age
np.quantile(data['age'], 0.75)

#Return counts of unique values.
data['state_of_res'].value_counts()

