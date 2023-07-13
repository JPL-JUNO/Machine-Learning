"""
@Description: Project one – predicting the fuel efficiency of a car
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-13 10:29:01
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv('auto-mpg.data',
                 names=column_names,
                 na_values='?',
                 comment='\t', sep=" ", skipinitialspace=True)
df = df.dropna().reset_index(drop=True)

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, train_size=.8, random_state=1)
train_stats = df_train.describe().transpose()
numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower',
                        'Weight', 'Acceleration']

df_train[numeric_column_names].sub(
    df_train[numeric_column_names].mean(), axis=1)
