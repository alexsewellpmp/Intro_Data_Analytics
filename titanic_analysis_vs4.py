# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 07:54:58 2016

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = ('C:\Users\Alex\Documents\Udacity\Data Analyst'
            r'\titanic_analysis\titanic_data.csv')

titanic_df = pd.read_csv(file_path)

categories = titanic_df.dtypes[titanic_df.dtypes == "object"].index

def data_review(data):
    '''Returns data for initial review to help understand shape, 
    content and basic statistics about the DataFrame before exploration'''
    print categories # Provides index of variables / column headers
    print titanic_df.shape # Details the data structure
    print titanic_df.head() # Displays first 5 rows of data
    print titanic_df.info() # Details the data types by variable
    print titanic_df.describe() # Describes quantifiable data stats
    
    return data_review

#print data_review(titanic_df)

''' Data Cleaning Phase - print statements validated new variable assignment'''

# Replaces 177 NaN 'Age' values with the median of 28
new_age_var = np.where(titanic_df["Age"].isnull(),
                       28,
                       titanic_df["Age"])

titanic_df["Age"] = new_age_var 

# Defines data set to be removed from the DataFrame.
titanic_df = titanic_df.drop(['Ticket','Cabin'], axis=1)
# Omits NaN values
titanic_df = titanic_df.dropna()

'''Exploration Phase - Uses both seaborn and matplotlib for visualizations'''

# Plot style specifications for visualizations
sns.set(font='verdana', style='whitegrid')

# Grouped Visualizations 
# Sets plot dimensions and specifications
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# Enables grouping of different plot types together 
ax1 = plt.subplot2grid((2,3),(0,0))
# Bar graph plot of survivors and non-survivors.               
titanic_df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# Applies title to the graph
plt.title("Mortality Counts, (1 = Survived)")    

plt.subplot2grid((2,3),(0,1))
plt.scatter(titanic_df.Survived, titanic_df.Age, alpha=alpha_scatterplot)
# Applies y axis lable
plt.ylabel("Age in Years")
plt.title("Mortality Distributed by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
titanic_df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
# Graph layout specifications
ax3.set_ylim(-1, len(titanic_df.Pclass.value_counts()))
plt.title("Passenger Class Counts")

# Allows density plot to span 2 positions
plt.subplot2grid((2,3),(1,0), colspan=2)
# Kernel density estimate subset 'Age' of by passangers class
titanic_df.Age[titanic_df.Pclass == 1].plot(kind='kde')    
titanic_df.Age[titanic_df.Pclass == 2].plot(kind='kde')
titanic_df.Age[titanic_df.Pclass == 3].plot(kind='kde')
# Applies x and y axis lables for the plot
plt.xlabel("Age in Years")
plt.ylabel("Density")
plt.title("Passenger Class Ages Comparison")
# Positions the legend with 'PClass' naming in respective order.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
titanic_df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# Graph layout specifications
ax5.set_xlim(-1, len(titanic_df.Embarked.value_counts()))
plt.title("Boarding Location Passenger Counts")


# Sets plot size for Survived variable subset 'Sex' graphs
fig = plt.figure(figsize=(18,6))
alpha=0.75


# Plots 2 subsets, male and female, by the variable 'Survived'.
df_male = titanic_df.Survived[titanic_df.Sex == 'male'].value_counts().sort_index()
df_female = titanic_df.Survived[titanic_df.Sex == 'female'].value_counts().sort_index()

# Calls value_counts() for the bar graph plotting. 
# 'barh' is to create a horizontal bar graph
ax1 = fig.add_subplot(121)
df_male.plot(kind='barh', color='#5D5EC7', label='Male', alpha=alpha)
df_female.plot(kind='barh', color='#CB4343',label='Female', alpha=alpha)
plt.title("Count of Survivors by Gender"); plt.legend(loc='best')
plt.xlabel("Raw Counts")
ax1.set_ylim(-1, 2) 

# Displays proportion of survivors by gender.
ax2 = fig.add_subplot(122)
(df_male/float(df_male.sum())).plot(kind='barh', color='#5D5EC7', label='Male', alpha=alpha)  
(df_female/float(df_female.sum())).plot(kind='barh', color='#CB4343',label='Female', alpha=alpha)
plt.title("Proportional Count of Survivors by Gender"); 
plt.legend(loc='best')
plt.xlabel("Proportion")
ax2.set_ylim(-1, 2)

