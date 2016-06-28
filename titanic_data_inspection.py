# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:29:46 2016

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='calibri light', style='whitegrid')


# Load the dataset
file_path = ('C:\Users\Alex\Documents\Udacity\Data Analyst'
            r'\titanic_analysis\titanic_data.csv')

titanic_df = pd.read_csv(file_path)
categories = titanic_df.dtypes[titanic_df.dtypes == "object"].index

print titanic_df['Survived'].describe

