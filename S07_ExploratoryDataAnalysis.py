#Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Read data from file
data = pd.read_csv('cs_01.csv')
data.head()

data.shape
data.info()

np.sum(data.isnull(), axis = 0)

#sex (categorical - binary), Check for imbalanced data
sex_freq = data['sex'].value_counts()
sex_freq

#Bar plot for sex
plt.bar(sex_freq.index, sex_freq.values)
plt.xticks(sex_freq.index)
plt.title('Dist. of Sex')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()

#age (numerical)
data['age'].describe()

#Histogram for age
plt.hist(data['age'], bins = np.arange(data['age'].min(), 
                                       data['age'].max()),
         color = 'red', alpha = 0.7)
plt.axvline(data['age'].mean(), 
            color = 'black', linewidth = 2, 
            linestyle = '--', label = "Average")
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#p_status (categorical - binary), check for imbalanced data
p_status_freq = data['p_status'].value_counts()
p_status_freq

#Bar plot for cohabitation status
plt.bar(p_status_freq.index, p_status_freq.values)
plt.xticks(ticks = [0, 1], labels = ['Living together', 'Apart'])
plt.title("Dist. of Parent's Cohabitation Status")
plt.xlabel("Parent's Cohabitation Status")
plt.ylabel('Frequency')
plt.show()

#m_edu (categorical - ordinal)
m_edu_freq = data['m_edu'].value_counts()
m_edu_freq

#Bar plot for m_edu
plt.bar(m_edu_freq.index, m_edu_freq.values)
plt.xticks(m_edu_freq.index)
plt.title("Dist. of Mother's Education")
plt.xlabel("Mother's Education")
plt.ylabel('Frequency')
plt.show()

#m_job (categorical - nominal)
m_job_freq = data['m_job'].value_counts()
m_job_freq

#Bar plot for m_job
plt.bar(m_job_freq.index, m_job_freq.values)
plt.xticks(m_job_freq.index)
plt.title("Dist. of Mother's Job")
plt.xlabel("Mother's Job")
plt.ylabel('Frequency')
plt.show()

#absences (numerical)
data['absences'].describe()

#Histogram for absences
plt.hist(data['absences'], bins = 15,
         color = 'red', alpha = 0.7)
plt.axvline(data['absences'].mean(), 
            color = 'black', linewidth = 2, 
            linestyle = '--', label = "Average")
plt.title('Histogram of Number of School Absences')
plt.xlabel('# of Absences')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#Box Plot
plt.boxplot(data['absences'])
plt.title('Boxplot of # of Absences')
plt.xticks([]) #remove the xticks 
plt.ylabel('# of Absences')
plt.show()

#final_grade (numerical)
data['final_grade'].describe()

#Histogram for final_grade
plt.hist(data['final_grade'], bins = 15,
         color = 'red', alpha = 0.7)
plt.axvline(data['final_grade'].mean(), 
            color = 'black', linewidth = 2, 
            linestyle = '--', label = "Average")
plt.title('Histogram of Final Grade')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#The number of students with final grade equals 0
np.sum(data['final_grade'] == 0)

#Histogram for final_grade (greater than zero)
positive_grade = data.loc[data['final_grade'] > 0, :]
plt.hist(positive_grade['final_grade'], bins = 15,
         color = 'red', alpha = 0.7)
plt.axvline(positive_grade['final_grade'].mean(), 
            color = 'black', linewidth = 2, 
            linestyle = '--', label = "Average")
plt.title('Histogram of Final Grade')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#Check for normality of final_grade

#Step 1: Histogram w/ density plot
#Calculate density
from scipy import stats
density = stats.gaussian_kde(positive_grade['final_grade'])
xp = np.linspace(positive_grade['final_grade'].min(), 
                 positive_grade['final_grade'].max(), 100)
yp = density.pdf(xp)

#Histogram
plt.hist(positive_grade['final_grade'], bins = 10, 
         color = 'red', alpha = 0.7, density = True)
plt.axvline(positive_grade['final_grade'].mean(), 
            color = 'black', linewidth = 2, 
            linestyle = '--', label = "Average")
plt.title('Histogram of Number of School Absences')
plt.xlabel('# of Absences')
plt.ylabel('Density')
#Add pdf curve
plt.plot(xp, yp, color = 'black', linewidth = 2)
plt.legend()
plt.show()

#Step 2: qq-plot
sm.qqplot(positive_grade['final_grade'], line = 's')
plt.show()
#Conclusion: data is normally distributed.

#Two continuous variables: final grade vs. absences
#Scatter plot
plt.scatter(x = positive_grade['absences'], 
            y = positive_grade['final_grade'], 
            color = 'red', alpha = 0.7)
plt.title('Final Grade vs. Absences')
plt.xlabel('Absences', fontsize = 12)
plt.ylabel('Final Grade', fontsize = 12)
plt.grid()
plt.show()

#Pearson correlation
positive_grade[['final_grade', 'absences']].corr(method = 'pearson')

#Spearman correlation
positive_grade[['final_grade', 'absences']].corr(method = 'spearman')

#Numerical vs categorical variables: final grade vs. p_status
positive_grade.groupby('p_status')['final_grade'].mean()

#Boxplot of final grade vs. p_status
plt.boxplot([positive_grade.loc[positive_grade['p_status'] == 'T', 
                                'final_grade'], 
             positive_grade.loc[positive_grade['p_status'] == 'A', 
                                'final_grade']])
plt.title("Boxplot of \n Final Grade vs. Parent's Cohabitation Status")
plt.xticks(ticks = [1, 2], labels = ['Living together', 'Apart'])
plt.xlabel("Parent's Cohabitation Status")
plt.ylabel('Final Grade')
plt.show()

#Numerical vs categorical variables: final grade vs. higher
positive_grade.groupby('higher')['final_grade'].mean()

#Boxplot of final grade vs. higher
plt.boxplot([positive_grade.loc[positive_grade['higher'] == 'yes', 
                                'final_grade'], 
             positive_grade.loc[positive_grade['higher'] == 'no', 
                                'final_grade']])
plt.title('Boxplot of \n Final Grade vs. Higher Education Intention')
plt.xticks(ticks = [1, 2], labels = ['Yes', 'No'])
plt.xlabel('Higher Education Intention')
plt.ylabel('Final Grade')
plt.show()

#Two categorical variables: Cross Tabulation Analysis
#fam_sup vs. m_edu
cross_tab = pd.crosstab(positive_grade['fam_sup'], 
                        positive_grade['m_edu'])
cross_tab

#Normalize over each column
cross_tab_pct = round(pd.crosstab(positive_grade['fam_sup'], 
                                  positive_grade['m_edu'], 
                                  normalize = 'columns'), 2)
cross_tab_pct

