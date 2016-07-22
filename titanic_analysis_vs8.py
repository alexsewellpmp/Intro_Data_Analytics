# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 07:54:58 2016

@author: Alex
"""

import numpy as np 
import pandas as pd # Enables data to be managed as dataframes
import matplotlib.pyplot as plt # Establishes plotting under plt
import seaborn as sns # Enhances aesthetics and plotting options


# Plot style specifications for visualizations
sns.set(font='verdana', style='whitegrid')


# Load the dataset
file_path = ('C:\Users\Alex\Documents\Udacity\Data Analyst'
             r'\titanic_analysis\titanic_data.csv')

titanic_df = pd.read_csv(file_path)


categories = titanic_df.dtypes[titanic_df.dtypes == 'object'].index
# print categories

def data_review(data):
    '''Returns data for initial review to help understand shape,
    content and basic statistics about the DataFrame before exploration'''
    print categories  # Provides index of variables / column headers
    print titanic_df.shape  # Details the data structure
    print titanic_df.head()  # Displays first 5 rows of data
    print titanic_df.info()  # Details the data types by variable
    print titanic_df.describe()  # Describes quantifiable data stats

    return data_review

# print data_review()

'''Data Cleaning and Transformations'''


def man_woman_child(passenger):
    '''Using Titanic's Certificate of Clearance (MT 9/920f), this assigns man 
    or woman to passengers aged 13 years or older given age and sex, child to 
    those 12 years of age and younger.'''
    Age, Sex = passenger
    if Age < 13:
        return 'Child'
    else:
        return dict(male='Man', female='Woman')[Sex]


# Cleans - Replaces 177 NaN 'Age' values with the median of 28
new_age_var = np.where(titanic_df['Age'].isnull(),
                       28,
                       titanic_df['Age'])

titanic_df['Age'] = new_age_var

# Cleans - Defines data set to be removed from the DataFrame.
titanic_df = titanic_df.drop(['Name', 'Ticket','Cabin'], axis=1)

# Cleans - Omits NaN values
titanic_df = titanic_df.dropna()

# Assigns a new variable 'Mortality' to represent 'Survived'.
# '0' and '1' are renamed 'Died' and 'Lived' respectively.
titanic_df['Mortality'] = titanic_df.Survived.map({0: 'Died', 1: 'Lived'})

# Transforms - Plots 2 subsets, male and female, by the variable 'Survived'.
df_male = (
    titanic_df.Mortality[titanic_df.Sex == 'male'].value_counts().sort_index())
df_female = (
    titanic_df.Mortality[titanic_df.Sex == 'female'].value_counts().
    sort_index())

# Transforms - Creates a new variable 'Person' using 'Age' and 'Sex' 
# then calls the man_woman_child function to assign the values 'Man',
# 'Woman', 'Child'
titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(man_woman_child,
axis=1)


'''Exploration Phase - Uses both seaborn and matplotlib for visualizations'''


## Univariate or 1D Analysis and Visualizations

# Presents boxplot for age distribution among passengers.
Age_1DChart = sns.boxplot('Age', data=titanic_df)
Age_1DChart.set(xlabel='Age Range')
Age_1DChart.axes.set_title('Age Range of Passengers', fontsize = 15,
                          color="black")
plt.show(Age_1DChart)

# Presents column chart of passenger counts by 'Sex' or gender.
Sex_1DChart = sns.countplot(x='Sex', data=titanic_df)
Sex_1DChart.set(xlabel='Passenger Gender', ylabel='Count of Passengers')
Sex_1DChart.axes.set_title('Passenger Count by Gender', fontsize = 15,
                          color="black")
plt.show(Sex_1DChart)

# Presents column chart of passenger counts by 'Survived'.
Mortality_1DChart = sns.countplot(x='Mortality', data=titanic_df)
Mortality_1DChart.set(xlabel='Mortality', ylabel='Count of Passengers')
Mortality_1DChart.axes.set_title('Passenger Mortality Count', fontsize = 15,
                          color="black")
plt.show(Mortality_1DChart)


## Bivariate or 2D Analysis and Visualizations
generations = [10, 20, 30, 40, 50, 60, 80]

Survival_Age_2DChart = sns.lmplot('Age', 'Survived', data=titanic_df)
Survival_Age_2DChart.ax.set_title('Survivability by Age', fontsize = 15)
plt.show(Survival_Age_2DChart)

Survival_Sex_2DChart = sns.countplot('Mortality', data=titanic_df, hue = 'Sex')
Survival_Sex_2DChart.set(xlabel='Sex', ylabel='Passenger Count')
Survival_Sex_2DChart.set_title('Survivors by Sex', fontsize = 15)
plt.show(Survival_Sex_2DChart)

Person_2DChart = sns.countplot(x='Person', data=titanic_df)
Person_2DChart.set(xlabel='Person Category',
               ylabel='Count of Passengers')
Person_2DChart.axes.set_title('Passenger Count by Category',
                          fontsize = 15,
                          color="black")
plt.show(Person_2DChart)



## Multivariate or 3D Analysis and Visualizations

Plt_Age_Sex_Survive = sns.lmplot('Age', 'Survived', data = titanic_df,
                                hue = 'Sex')
Plt_Age_Sex_Survive.ax.set_title('Survivability by Age and Sex', fontsize = 15)
plt.show(Plt_Age_Sex_Survive)



Person_Survival = sns.countplot(y='Mortality', hue='Person', data=titanic_df)
Person_Survival.set(xlabel='Count of Persons', ylabel='Personal Category')
Person_Survival.axes.set_title('Adult to Child Survival Counts', fontsize = 15,
                               color="black")
plt.show(Person_Survival)

Survive3D = sns.lmplot('Age','Survived',hue = 'Sex',data = titanic_df,
                           x_bins = generations)
plt.show(Survive3D)                 
                           


