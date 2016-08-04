# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 07:54:58 2016

@author: Alex
"""

import numpy as np  # Enables scientific mathmatical calculations
import pandas as pd  # Enables data to be managed as dataframes
import matplotlib.pyplot as plt  # Establishes plotting under plt
import seaborn as sns  # Enhances aesthetics and plotting options


''' Initial Data Handling and Review '''

# Plot style specifications for visualizations
sns.set(font='verdana', style='whitegrid')
sns.set_context(rc={"figure.figsize": (8, 4)})

# Load the dataset
file_path = ('C:\Users\Alex\Documents\Udacity\Data Analyst'
             r'\titanic_analysis\titanic_data.csv')
# Read the dataset
titanic_df = pd.read_csv(file_path)

# Produces column or variable overview with counts of rows.
pd.DataFrame({'Rows': titanic_df.count()}).T


def data_review(data):
    '''Returns data for initial review to help understand shape,
    content and basic statistics about the DataFrame before exploration'''
    print titanic_df.shape  # Details the data structure
    print titanic_df.head()  # Displays first 5 rows of data
    print titanic_df.info()  # Details the data types by variable
    print titanic_df.describe()  # Describes quantifiable data stats

    return data_review

# print data_review()

'''Data Cleaning and Transformations'''

# Cleans - Replaces 177 NaN 'Age' values with the median of 28
new_age_var = np.where(titanic_df['Age'].isnull(),
                       28,
                       titanic_df['Age'])

# Cleans - Appends median age to 'Age' variable
titanic_df['Age'] = new_age_var

# Cleans - Defines data set to be removed from the DataFrame.
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Cleans - Omits NaN values
titanic_df = titanic_df.dropna()

# Transforms - Creates new variable 'Mortality' to represent 'Survived'.
# '0' and '1' are renamed 'Died' and 'Lived' respectively.
titanic_df['Mortality'] = titanic_df.Survived.map({0: 'Died', 1: 'Lived'})

Age_Scale = [10, 20, 30, 40, 50, 60, 80]

# Transforms - Creates new variable 'Class' to represent 'PClass'.
# '1' '2' and '3' are renamed 'First', 'Second' and 'Third' respectively.
titanic_df['Class'] = titanic_df.Pclass.map({1: 'First',
                                             2: 'Second',
                                             3: 'Third'})


'''UNIVARIATE ANALYSIS'''

# Presents boxplot for age distribution among passengers.
Age_1Dboxplot = sns.boxplot('Age', data=titanic_df)
Age_1Dboxplot.set(xlabel='Age Range')
Age_1Dboxplot.axes.set_title(
    'Age Range of Passengers', fontsize=15, color="black")
plt.show(Age_1Dboxplot)

# Presents bar chart of raw passenger counts by 'Sex'.
Sex_1DCountplot = sns.countplot(x='Sex', data=titanic_df)
Sex_1DCountplot.set(xlabel='Passenger Gender', ylabel='Count of Passengers')
Sex_1DCountplot.axes.set_title(
    'Passenger Count by Gender', fontsize=15, color="black")
plt.show(Sex_1DCountplot)

# Presents bar chart of raw passenger counts by 'Survived'.
# Uses new 'Mortality' variable instead of 'Survived'
Mortality_1DCountplot = sns.countplot(x='Mortality', data=titanic_df)
Mortality_1DCountplot.set(xlabel='Mortality', ylabel='Count of Passengers')
Mortality_1DCountplot.axes.set_title(
    'Passenger Mortality Count', fontsize=15, color="black")
plt.show(Mortality_1DCountplot)

# Calculates holistic survival percentage based on the raw counts.
survival_mean = titanic_df['Survived'].mean()
print 'Count of Survived is {} out of {}'.format(
    titanic_df['Survived'].sum(), titanic_df['Survived'].count())
print "Average Survival: {}".format(round(survival_mean, 2))

# Presents bar chart of raw passenger counts by using the new variable 'Class'.
Pclass_1DCountplot = sns.countplot(x='Class', data=titanic_df)
Pclass_1DCountplot.set(xlabel='Passenger Class', ylabel='Count of Passengers')
Pclass_1DCountplot.axes.set_title(
    'Passenger Count by Class', fontsize=15, color="black")
plt.show(Pclass_1DCountplot)

'''BIVARIATE ANALYSIS'''

# Presents a linear regression plot relation 'Age' to 'Survived'
# Trends survivability by 'Age'
Survival_Age_lmplot = sns.lmplot('Age', 'Survived', data=titanic_df)
Survival_Age_lmplot.ax.set_title('Trend of Survival by Age', fontsize=15)
plt.show(Survival_Age_lmplot)

# Presents a violin plot of passenger class age density.
# Uses new variable 'Class' in place of 'Pclass'
Age_Class_Violin = sns.violinplot('Class', 'Age',
                                  data=titanic_df)
Age_Class_Violin.axes.set_title('Age Distribution by Class',
                                fontsize=15,
                                color='black')
plt.show(Age_Class_Violin)

# Presents a violin plot of 'Mortality' 'Age' density.
Age_Sex_Violin = sns.violinplot('Sex', 'Age',
                                data=titanic_df)
Age_Class_Violin.axes.set_title('Age Distribution by Class',
                                fontsize=15,
                                color='black')
plt.show(Age_Class_Violin)

# Presents a violin plot of 'Age' distribution of those that 'Lived' or 'Died'.
Survival_Age_Violin = sns.violinplot('Mortality', 'Age', data=titanic_df,
                                     cut=0)
Survival_Age_Violin.set_title('Survivability Age Distribution', fontsize=15)
plt.show(Survival_Age_Violin)

# Presents a bar chart of raw counts showing 'Mortality' by 'Sex'
Survival_Sex_Counts = sns.countplot('Mortality', data=titanic_df, hue='Sex')
Survival_Sex_Counts.set(xlabel='Sex', ylabel='Passenger Count')
Survival_Sex_Counts.set_title('Survivors by Sex', fontsize=15)
plt.show(Survival_Sex_Counts)

# Presents a pie chart of female proportional survival
# Proportion of women who survived is 0.742038216561
PropSurvival_FemPie = (
    titanic_df.Mortality[titanic_df.Sex == 'female'].value_counts() /
    float(titanic_df.Sex[titanic_df.Sex == 'female'].size)).plot(
        kind='pie',
        title='Female',
        autopct='%1.1f%%')
plt.show(PropSurvival_FemPie)

# Presents a pie chart of female proportional survival
# Proportion of men who survived is 0.188908145581
PropSurvival_MalePie = (
    titanic_df.Mortality[titanic_df.Sex == 'male'].value_counts() /
    float(titanic_df.Sex[titanic_df.Sex == 'male'].size)).plot(
        kind='pie',
        title='Male',
        autopct='%1.1f%%')
plt.show(PropSurvival_MalePie)

'''MULTIVARIATE ANALYSIS'''

# Presents a violin plot of 'Mortality' 'Age' density by Class
Class_Age_Mortality_Violin = sns.violinplot('Mortality', 'Age',
                                            data=titanic_df, hue='Class')
Class_Age_Mortality_Violin.legend(loc='best')
Class_Age_Mortality_Violin.axes.set_title('Class Age Mortality Distribution',
                                          fontsize=15,
                                          color='black')
plt.show(Class_Age_Mortality_Violin)

# Presents a bar chart of raw counts of passengers by 'Class' and 'Sex'
Survival_Sex_Counts = sns.countplot('Sex', data=titanic_df, hue='Class')
Survival_Sex_Counts.set(xlabel='Sex', ylabel='Passenger Count')
Survival_Sex_Counts.set_title('Male to Female Counts Per Class', fontsize=15)
plt.show(Survival_Sex_Counts)

# Presents a bar chart of raw counts of passengers by 'Class' and 'Mortality'
Survival_Sex_Counts = sns.countplot('Mortality', data=titanic_df,
                                    hue='Class')
Survival_Sex_Counts.set(xlabel='Class', ylabel='Passenger Count')
Survival_Sex_Counts.set_title('Survivors by Passenger Class', fontsize=15)
plt.show(Survival_Sex_Counts)

# Presents a linear regression plot of 'Survived' by 'Age' and 'Sex'
Age_Sex_Survive = sns.lmplot('Age', 'Survived', data=titanic_df, hue='Pclass')
plt.show(Age_Sex_Survive)

# Presents a linear regression plot of 'Survived' by 'Age' and 'Sex'
# Displays age increasing in increments of 10 and groups them
Survive3D = sns.lmplot(
    'Age', 'Survived', hue='Sex', data=titanic_df, x_bins=Age_Scale)
plt.show(Survive3D)

