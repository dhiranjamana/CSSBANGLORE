
# coding: utf-8

# In[29]:

import numpy as np
# import plotly.graph_objs as go
import pandas as pd
import glob

path ="C:/Users/css112831/Desktop/GoogleTrends/France"
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
    
df_fr = pd.concat(list_, axis=1)
df_france = df_fr.loc[:,~df_fr.columns.duplicated()]

path ="C:/Users/css112831/Desktop/GoogleTrends/Mexico"
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
    
df_Me = pd.concat(list_, axis=1)
df_Mexico = df_Me.loc[:,~df_Me.columns.duplicated()]

path ="C:/Users/css112831/Desktop/GoogleTrends/UK"
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
    
df_uk = pd.concat(list_, axis=1)
df_UK = df_uk.loc[:,~df_uk.columns.duplicated()]

path ="C:/Users/css112831/Desktop/GoogleTrends/USA"
allFiles = glob.glob(path + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)

df_us = pd.concat(list_, axis=1)
df_usa = df_us.loc[:,~df_us.columns.duplicated()]

df_france.to_csv("C:/Users/css112831/Desktop/GoogleTrends/France/Combined_fr.csv")
df_Mexico.to_csv("C:/Users/css112831/Desktop/GoogleTrends/Mexico/Combined_mexico.csv")
df_UK.to_csv("C:/Users/css112831/Desktop/GoogleTrends/UK/Combined_uk.csv")
df_usa.to_csv("C:/Users/css112831/Desktop/GoogleTrends/USA/Combined_usa.csv")

data = pd.read_csv("C:/Users/css112831/Desktop/Apps/Vaxitrend_state_combined/data_address_normalized.csv")


Mexico_states_data = list(data["administrative_area_level_1"][data["country"]=="Mexico"].unique())
len(Mexico_states_data)
US_states_data = list(data["administrative_area_level_1"][data["country"]=="United States"].unique())
len(US_states_data)
UK_states_data = list(data["administrative_area_level_1"][data["country"]=="United Kingdom"].unique())
len(UK_states_data)
France_states_data = list(data["administrative_area_level_1"][data["country"]=="France"].unique())
len(France_states_data)

path ="C:/Users/css112831/Desktop/GoogleTrends/USA/Geomap"
allFiles = glob.glob(path + "/*.csv")
US = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    US.append(df["Category: All categories"])
    
result = pd.concat(US,axis=1)
result1 = result.fillna(0)

result1.to_csv("C:/Users/css112831/Desktop/GoogleTrends/USA/GoogleTrends.csv")

path ="C:/Users/css112831/Desktop/GoogleTrends/UK/Geomap"
allFiles = glob.glob(path + "/*.csv")
UK = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    UK.append(df["Category: All categories"])
result = pd.concat(UK,axis=1)
result1 = result.fillna(0)

result1.to_csv("C:/Users/css112831/Desktop/GoogleTrends/UK/GoogleTrends_UK.csv")    
    
path ="C:/Users/css112831/Desktop/GoogleTrends/Mexico/Geomap"
allFiles = glob.glob(path + "/*.csv")
Mexico = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    Mexico.append(df["Category: All categories"])
result = pd.concat(Mexico,axis=1)
result1 = result.fillna(0)

result1.to_csv("C:/Users/css112831/Desktop/GoogleTrends/Mexico/GoogleTrends_Mexico.csv")  

path ="C:/Users/css112831/Desktop/GoogleTrends/France/Geomap"
allFiles = glob.glob(path + "/*.csv")
France = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    France.append(df["Category: All categories"])
result = pd.concat(France,axis=1)
result1 = result.fillna(0)

result1.to_csv("C:/Users/css112831/Desktop/GoogleTrends/France/GoogleTrends_France.csv")    
    
us = pd.read_csv("C:/Users/css112831/Desktop/GoogleTrends/USA/GoogleTrends_US.csv")
uk = pd.read_csv("C:/Users/css112831/Desktop/GoogleTrends/UK/GoogleTrends_UK.csv")  
fr = pd.read_csv("C:/Users/css112831/Desktop/GoogleTrends/Mexico/GoogleTrends_Mexico.csv") 
mex = pd.read_csv("C:/Users/css112831/Desktop/GoogleTrends/France/GoogleTrends_France.csv")

trends = us.append([uk,fr,mex])
trends1 = trends.fillna(0)

trends1.to_csv("C:/Users/css112831/Desktop/GoogleTrends/Gtrends_allstates.csv",index = False)

