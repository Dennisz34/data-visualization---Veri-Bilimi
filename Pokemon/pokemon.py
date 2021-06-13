# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:47:45 2021

@author: Casper
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output


data = pd.read_csv("pokemon.csv")


data.head()
print(data.columns)
#Satırdaki değerleri gösterir.
print(data.shape)
#Satır ve sütunların kaç tane olduğunu gösterir.
data.describe()
#Data hakkında birçok işlem yapar.
print(data.info())
#Datanın bilgisini verir.

data.isnull().sum()
#Null değere sahip verilerin sayısını verir.


#%%Exploratory data analysis (EDA)

print(data['Type 1'].value_counts(dropna =False))

b = data.corr()

a = data.describe()

#correlation map
f,ax =plt.subplots(figsize =(7,5))
sns.heatmap(data.corr(), annot=True, linewidths=.5,fmt =".1f",ax=ax)
plt.show()

#%%
#Visual exploratory data analysis

data.boxplot(column="Attack",by="Legendary")
plt.show()


#%%Tidy data

data_new = data.head()
#Yeni bir data oluşturduk ve veri setimizdeki ilk 5 veriyi içerisine attık.

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted

melted2 = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Speed','HP'])
melted2

#%% Pivoting data
print(melted.pivot(index ="Name",columns="variable",values="value"))

print(melted2.pivot(index ="Name",columns="variable",values="value"))


#%%Concatenating data
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1, data2], ignore_index=True,axis=0)

data3 = data["Attack"].head()
data4 = data["Defense"].head()


conc_data_col = pd.concat([data1,data2],axis=1)
print(conc_data_col)


#Data types
#%% Missing data and testing with assert

print(data["Type 2"].value_counts(dropna=False))

print(data["Type 2"].value_counts(dropna=True))
#Nan olan değerleri göstermez.

data1["Type 2"].dropna(inplace =True)



data1["Type 2"].fillna('Pok-emon',inplace = True)

#%%
#Index objects and labeled data
print(data.index.name)

print(data.index.name)


data.index.name = "index_name"
print(data.head())


data3 = data.copy()
data3.index = range(100,900,1)
print(data3.head())


#%% Hierarchical indexing

data1 = data.set_index

data1 = data.set_index(["Type 1","Type 2"])
print(data1.head())

#%% Pivoting data frames

# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')

#%% Concatenating data
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 
# axis = 0 : adds dataframes in row
conc_data_row


data1 = data['Attack'].head()
data2= data['Speed'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col


print(data.dtypes)
#data['Type 1'] = data['Type 1'].astype('category')
#data['Speed'] = data['Speed'].astype('float')
#print(data.dtypes)

"""
#Manipulating Data Frames with Pandas

# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
data= data.set_index("#")
# also you can use 
# data.index = data["#"]
print(data.index.name)
data.index.name = "index_name" # lets change it
data.head()

"""


# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to orok then change index 
orok = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
orok.index = range(100,900,1)
orok.head()




#Hierarchical indexing
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Type 1","Type 2"]) 
data1.head(10)
# data1.loc["Fire","Flying"] # howw to use indexes

#%%Pivoting data frames

dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df






































































