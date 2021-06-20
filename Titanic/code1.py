# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:06:45 2021

@author: Casper
"""

#titanic 1

"""
The sinking of the world's largest ship in 1912
"""

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

print(train_df.columns)


# PassengerId = yolcuların numarası
# Survived = Hayatta kalan,0= öldü,1=hayatta kalmıs
# SibSp = akraba sayısı
# Parch = gemi içerisinde ailesi varmı yokmu
# Ticket = bilet numarası
# Fare = bilete ödediğ para mıktarı
# Embarked = hangı limandan bındıgı
#Pclass = yolcuların sınıfları
# Sex = cinsiyet
# Age = yas
# SibSp = Kardeşlerin / Eşlerin Sayısı
# Parch = Ebeveynlerin / çocuk sayısı


print()

a = (train_df.head())

print(a)

print()

b = (train_df.describe())

print(train_df.info())


#train_df.drop(['PassengerId','Ticket','Name'],inplace=True,axis=1)  #Droping few features

table = pd.pivot_table(data=train_df,index=['Sex']) # ‘Sex’ column as the index,single index
table

table.plot(kind = "bar")



#--Categorical Variable

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["Fare", "Age","PassengerId"]

for n in numericVar:
    plot_hist(n)


#%%


def bar_plot:
    var = train_df[variable]
    # Kategorik Değişken Sayısı (Değer / Örnek)
    varValue = var.value_counts()
    print("{}: \n {} ".format(variable , varValue))
    

category1 = ["Surviveved" , "Sex" , "Pclass" , "Embarked" , "SibSp" , "Parch"]
for c in category1:
        bar_plot(c)


#%%

def bar_plot:
        
    """     
     input: variable ex: "Sex"
     output: bar plot & value count
    """
    #Özelliiğini almak
    var = train_df[variable]
    # Kategorik Değişken Sayısı (Değer / Örnek)
    varValue = var.value_counts()
    #cinsiyetimizden kaç tane olması gerektiğini sağlıyor
    
    # görselleştirmek
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index , varValue)
    plt.xticks(varValue.index , varValue.index.value)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    
category2 = ["Cabin" , "Name" , "Ticket"]

for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


    
#%% Numerical Variable

def plot_hist(variable):
    plt.figure(figsize =  (9,3))
    plt.hist(train_df[variable] , bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
    
numericVar = ["Fare" , "Age" , "PassengerId"]

for i in numericVar:
    plot_hist(n)








































