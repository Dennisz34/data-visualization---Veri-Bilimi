# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:33:54 2021

@author: Deniz_Uku
"""

# -Find Missing Value
# -Fill Missing Value


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


#train_df içerisindeki veri sayısı
train_df_len = len(train_df)
print(train_df_len)

#train_df ile test_df yi birleştirdik , indeksleri sıfırladık ve train_df nin içine attık
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)

print(train_df.head())


# Find Missing Value
#Eksik olan verileri bulma
print(train_df.columns[train_df.isnull().any()])
print("-"*30)

#Boş(nan) olan veri sayısını verir.
print(train_df.isnull().sum())
print("-"*30)



# Fill Missing Value
#Boş olan verileri sildik
b = (train_df[train_df["Embarked"].isnull()])

print(b)

train_df.boxplot(column="Fare",by = "Embarked")
plt.show()

print()

#Boş olan verilerin yerine C yazdırdım
train_df["Embarked"] = train_df["Embarked"].fillna("C")
print("-"*30)

train_df[train_df["Fare"].isnull()]

#Pclass ta 3 e eşit olan verileri çek ve d ye at
d = train_df[train_df["Pclass"] == 3]["Fare"]
#d deki verilerin ortalamasını al
print(np.mean(d))

#Fare datasında boş olan verilerin yerine d nin ortalaması yaz
train_df["Fare"]=train_df["Fare"].fillna(np.mean(d))

#fare datasında boş olan verileri göster.
print(train_df[train_df["Fare"].isnull()])



table = pd.pivot_table(train_df,index=['Sex','Pclass'],aggfunc={'Age':np.mean,'Survived':np.sum}) 
table

#aggfunc = toplama fonksiyonu age deki verinin ortalamasını al survived teki verileri ise topla
table =pd.pivot_table(train_df , index =["Sex" ,"Pclass"],aggfunc = {"Age":np.mean,"Survived":np.sum})

table = pd.pivot_table(train_df,index=['Sex','Pclass'],values=['Survived'], aggfunc=np.mean)
table


table.plot(kind='bar');
#%%

table = pd.pivot_table(train_df , index = ["Sex"] , columns = ["Pclass"] ,  values  = ["Survived"] , aggfunc = np.sum)
table

table.plot(kind = "bar")




table = pd.pivot_table(train_df,index=['Sex','Survived','Pclass'],columns=['Embarked'],values=['Age'],aggfunc=np.mean) #nullvalues
table


table.plot(kind="bar")

table = pd.pivot_table(train_df,index=['Sex','Survived','Pclass'],columns=['Embarked'],values=['Age'],aggfunc=np.mean,fill_value=np.mean(train_df['Age'])) #replacing the NaN values with the mean value from the ‘Age’ column
table



















