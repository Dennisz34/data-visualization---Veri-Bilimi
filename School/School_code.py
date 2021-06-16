# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 17:10:21 2021

@author: Casper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
# plotly
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt


data = pd.read_csv("cwurData.csv")
#data1 = pd.read_csv("education_expenditure_supplementary_data.csv")
data2 = pd.read_csv("educational_attainment_supplementary_data.csv")
data3 = pd.read_csv("school_and_country_table.csv")
data4 = pd.read_csv("shanghaiData.csv")
data5 = pd.read_csv("timesData.csv")

# information about timesData
data.info()


#data1.head(10)
data2.head(10)
data3.head(10)
data4.head(10)
data5.head(10)


data.tail()
#Son 5 veri

data.loc[0:50,"institution":"national_rank"] 
#0 ile 50 arası ülke ve doğal olanları getir.


data.groupby('institution').patents.sum().sort_values(ascending = True)
#Ülkelerin patent alma sayısını ver.

from wordcloud import WordCloud
 
uni1 = data.head(10)
plt.scatter(uni1.country,uni1.institution,)
plt.xlabel('countries')
plt.ylabel('universities')
plt.title('rank')
plt.show()




#%% Line Charts

df = data5.iloc[:100,:]

trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.university_name)


trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)


data=[trace1,trace2]
layout = dict(title ="Citation and Teaching vs World Rank of Top 100 Universities",
              xaxis=dict(title="World Rank",ticklen=5,zeroline =True)
              )

fig = dict(data=data,layout=layout)
plot(fig)


#%%

df2 = data5.iloc[:100,:]
trace3 = go.Scatter(
                    x = df2.world_rank,
                    y = df2.research,
                    mode = "lines",
                    name = "world_rank",
                    marker = dict(color = 'rgba(8, 12, 30, 0.8)'),
                    text= df.university_name)
                    
                    
trace4 =go.Scatter(
                    x = df2.teaching,
                    y = df2.research,
                    mode = "lines",
                    name = "teaching",
                    marker = dict(color = 'rgba(8, 12, 30, 0.8)'),
                    text= df.university_name) 
    
    
    
data =[trace3,trace4]
    

layout = dict(title ="Teaching and World rank vs research  of Top 100 Universities",
              xaxis=dict(title="research ",ticklen=5,zeroline =False))

fig = dict(data =data ,layout = layout)

plot(fig)


#%%
df2 = data5.iloc[:100,:]

a = go.Scatter(
    x = df2.world_rank,
    y = df2.research,
    mode = "lines",
    name = "world_rank",
    marker = dict(color = 'rgba(3, 111, 20, 0.9)'),
    text= df.university_name)


b =go.Scatter(
    x = df2.world_rank,
    y = df2.teaching,
    mode = "lines+markers",
    name = "teaching",
    marker = dict(color = 'rgba(8, 12, 30, 0.8)'),
    text= df.university_name) 



data =[a,b]


layout = dict(title ="Teaching and World rank vs research  of Top 100 Universities",
              xaxis=dict(title="research ",ticklen=5,zeroline =False))

fig = dict(data =data ,layout = layout)

plot(fig)




#%%
df2014 = data5[data5.year == 2014].iloc[:100,:]
df2015 = data5[data5.year == 2015].iloc[:100,:]
df2016 = data5[data5.year == 2016].iloc[:100,:]

trace1 =go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2014.university_name)
trace2 =go.Scatter(
                    x = df2015.world_rank,
                    y = df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2015.university_name)
#  trace3
trace3 =go.Scatter(
                    x = df2016.world_rank,
                    y = df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2016.university_name)

data = [trace1,trace2,trace3]

layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

plot(fig)


#%%

# Bar Charts

df2014 = data5[data5.year == 2014].iloc[:100,:]
df2015 = data5[data5.year == 2015].iloc[:100,:]
df2016 = data5[data5.year == 2016].iloc[:100,:]

df2014 = data5[data5.year == 2014].iloc[:3,:]
print(df2014)

# prepare data frames
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
# create trace2 
trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
plot(fig)

#%%

# Bar Charts

df2014 = data5[data5.year == 2014].iloc[:100,:]
df2015 = data5[data5.year == 2015].iloc[:100,:]
df2016 = data5[data5.year == 2016].iloc[:100,:]

df2014 = data5[data5.year == 2014].iloc[:3,:]
print(df2014)

# prepare data frames
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
# create trace2 
trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
plot(fig)


#%%

va = go.Bar(
        x = df2014.university_name,
        y = df2014.citations,
        name = "citations",
        marker = dict(color = 'rgba(100, 44, 55, 0.5)',
                      line=dict(color='rgb(0,0,0)',width=1.5)),
        text = df2014.country)
b = go.Bar(
        x = df2014.university_name,
        y = df2014.teaching,
        name = "teaching",
        marker = dict(color = 'rgba(65, 55, 122, 0.5)',
                      line=dict(color='rgb(0,0,0)',width=1.5)),
        text = df2014.country)

data = [a, b]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
plot(fig)




















































































































































































































































































































