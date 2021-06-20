# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:17:29 2021

@author: Casper
"""

# Basic Data Analysis

#Load and Check Data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_passengerId = test_df["PassengerId"]
print()


# Plcass vs Survived
train_df[["Pclass" , "Survived"]].groupby(["Pclass"] , as_index = False).mean().sort_values(by = "Survived" , ascending = False)
#Pclass'tan 3 tür var bu 3 türü Survived datalarına göre ortalamasını al ve pclass a göre grupla ==== ascending --> Yükselen , artan
print()


# Sex vs Survived
print(train_df[["Sex" , "Survived"]].groupby(["Sex"] , as_index = False).mean().sort_values(by = "Survived" , ascending = False))
print()



# Sibsp vs Survived
print(train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False))

print(train_df[["SibSp" , "Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by = "Survived" , ascending =True))
print()

# Parch vs Survived
print(train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False))

#Outlier Detection

def detect_outliers(df,features):
    outlier_indices = []

    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)































































