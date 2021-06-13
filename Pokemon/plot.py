# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 22:10:13 2021

@author: Casper
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

data = pd.read_csv("pokemon.csv")


# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()



data1 = (data[data["Attack"]>100].head())
data2 = (data[data["Speed"]>90].head())


data1.Attack.plot(kind="line",color ="blue",label="attack",
                  linewidth=1,alpha=0.5,grid=True,linestyle="-")

data2.Speed.plot(kind="line",color="red",label="speed",
                 linewidth=1,alpha=0.6,grid=True,linestyle=":")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()


#%%

data3 = (data[data["HP"]>100].head(20))
data4 = (data[data["Defense"]>100].head(20))


data3.HP.plot(kind = "line" , color="pink" , label = "Can" ,linewidth=1,alpha=0.5,grid=True,linestyle="-" )
data4.Defense.plot(kind  ="line" , color ="blue" , label = "Defans",linewidth=1,alpha=0.5,grid=True,linestyle="-")
plt.legend(loc='upper right') 
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()
#%%
# Scatter Plot 
# x = attack, y = defense
data.plot(kind = "scatter" , x ="Attack" , y = "Defense",alpha = 0.5,color = 'red',grid=True)
plt.xlabel('Atak')              # label = name of label
plt.ylabel('Savunma')
plt.title('Atak ve defans Scatter plot grafiği')            # title = title of plot

# x = HP, y = defense
data.plot(kind="scatter" , x = "HP" , y = "Defense",alpha = 0.5,color = 'red',grid=True)
plt.xlabel('Can')              # label = name of label
plt.ylabel('Savunma')
plt.title('Can ve defans Scatter plot grafiği') 

#%%
data5 = (data[data["HP"]>100].head(20))
data6 = (data[data["Attack"]>100].head(20))
data7 =pd.concat([data5,data6],axis=0)

data7.plot(kind="scatter",x="HP",y="Defense",alpha=0.9,
           color="red")

plt.xlabel("HP")
plt.ylabel("Attack")
plt.title("HP Attack Scatter Plot")
plt.show()
#%%
data8 = (data[data["HP"]>100].head(20))
data9 = (data[data["Attack"]>100].head(20))
data10 = pd.concat([data8,data9],axis=0)

data10.plot(kind ="scatter",x="HP" , y="Defense",alpha=0.9,
           color="red",grid=True)
plt.legend()
plt.xlabel("HP")
plt.ylabel("Defense")
plt.title("Ortaya karışık Scatter Plot")
plt.show()

#%%
# Histogram
# bins = number of bar in figure
data.Speed.plot(kind="hist" , bins=50 , figsize = (10,6))
plt.show()

data.Attack.plot(kind="hist" , bins =150 , figsize =(10,5) , color="red")
plt.show()

data.Defense.plot(kind="hist" , bins=150 , figsize=(10,5),color="pink",grid=True)
plt.show()

#%%

data.boxplot(column='Attack',by = 'Legendary')

# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# it is confusing


data2 = data.loc[:,["HP","Defense","Attack"]]
data2.plot()
#%%


from matplotlib import style

print(plt.style.available)

plt.style.use('Solarize_Light2')

# importing all the necessary packages
import numpy as np
import matplotlib.pyplot as plt

# importing the style package
from matplotlib import style

# creating an array of data for plot
data = np.random.randn(50)

# using the style for the plot
plt.style.use('Solarize_Light2')

# creating a plot
plt.plot(data)

# show plot
plt.show()


# subplots
data1.plot(subplots=True)
plt.show()


data2.plot(subplots =True,grid=True)
plt.style.use('grayscale')
plt.show()

#%%

fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),ax = axes[0],color="aqua")
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),ax = axes[1],cumulative = True,color="pink")
plt


#%% SLICING DATA FRAME
#dilimleme

a = data.loc[1:10,"HP":"Defense"]

b = data.loc[10:20,"HP":"Defense"]
print(b)


# From something to end
c = data.loc[1:10,"Speed":]

print(c)
print("-"*30)
# FILTERING DATA FRAMES

boolean =data.HP > 200

d  = (data[boolean])
print((data[boolean]))

# Combining filters

first_filter = data.HP>150
second_filter = data.Speed>40
al = data[first_filter & second_filter]
print(al.head())

print("-"*40)

# Filtering column based others
print(data.HP[data.Speed<15])


print(data.loc[495])

print(data.Speed[data.HP<100])

print("-"*30)

#TRANSFORMING DATA


def div(n):
    return n/2
print(data.HP.apply(div))

def aa(m):
    return m/2
print(data.Speed.apply(aa))
print("-"*30)


print(data.HP.apply(lambda n : n/2))

print(data.Attack.apply(lambda c : c/2))

data["total_power"] = data.Attack + data.Defense
print()
print(data.head())








