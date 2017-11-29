
# coding: utf-8

# In[101]:

import numpy as np
import pandas as pd
import scipy
import math


# In[102]:

df = pd.read_excel("C:/Users/css112831/Desktop/Google - Sanofi (Data Sources)/data/ttest_data.xlsx",sheet = "Sheet1")


# In[103]:

df.columns


# In[104]:

names = list(df.columns)
names = names[2:]


# In[105]:

names


# In[106]:

countries = ["Mexico","China","FR","US","UK"]


# In[107]:

# z = 1.96
# n1 = len(df[df["Diabetes"]==1])
# n2 = len(df)
# x1 = np.mean(df["Knowledge of Flu"][df["Diabetes"]==1])
# x2 = np.mean(df["Knowledge of Flu"])
# s1 = np.var(df["Knowledge of Flu"][df["Diabetes"]==1])
# s2 = np.var(df["Knowledge of Flu"])
# std_er = math.sqrt((s1/n1)+(s2/n2))
# r1 = abs(x1-x2) + (z * (std_er))
# r2 = abs(x1-x2) - (z * (std_er))
# print str(r1)+"    "+str(r2) + "    "+str(std_er)+ "    "+ str(abs(x1-x2))
 


# In[108]:

min_val = []
max_val = []
mean_diff = []
question = []
mean_dia = []
mean = []
country = []
z = 1.96
for j in countries:
    for i in names:
        n1 = len(df[(df["Diabetes"]==1) & (df["Country"]==j)])
        n2 = len(df[df["Country"]==j])
        x1 = np.mean(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        x2 = np.mean(df[i][df["Country"]==j])
        s1 = np.var(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        s2 = np.var(df[i][df["Country"]==j])
        std_er = math.sqrt((s1/n1)+(s2/n2))
        r1 = abs(x1-x2) + (z * (std_er))
        r2 = abs(x1-x2) - (z * (std_er))
        min_val.append(r1)
        max_val.append(r2)
        mean_diff.append(abs(x1-x2))
        question.append(i)
        mean_dia.append(x1)
        mean.append(x2)
        country.append(j)
result = pd.DataFrame({"Country":country,"Question":question,"Mean_diff":mean_diff,"Max":max_val,"Min":min_val,"Mean_dia":mean_dia,"Mean_Pop":mean})    


# In[109]:

result["Pop_max"] = result["Mean_Pop"] + result["Max"]
result["Pop_min"] = result["Mean_Pop"] + result["Min"]


# In[110]:

min_val = []
max_val = []
mean_diff = []
question = []
mean_dia = []
mean = []
country = []
z = 1.75
for j in countries:
    for i in names:
        n1 = len(df[(df["Diabetes"]==1) & (df["Country"]==j)])
        n2 = len(df[df["Country"]==j])
        x1 = np.mean(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        x2 = np.mean(df[i][df["Country"]==j])
        s1 = np.var(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        s2 = np.var(df[i][df["Country"]==j])
        std_er = math.sqrt((s1/n1)+(s2/n2))
        r1 = abs(x1-x2) + (z * (std_er))
        r2 = abs(x1-x2) - (z * (std_er))
        min_val.append(r1)
        max_val.append(r2)
        mean_diff.append(abs(x1-x2))
        question.append(i)
        mean_dia.append(x1)
        mean.append(x2)
        country.append(j)
result1 = pd.DataFrame({"Country":country,"Question":question,"Mean_diff":mean_diff,"Max":max_val,"Min":min_val,"Mean_dia":mean_dia,"Mean_Pop":mean})    


# In[111]:

result1["Pop_max"] = result1["Mean_Pop"] + result1["Max"]
result1["Pop_min"] = result1["Mean_Pop"] + result1["Min"]


# In[112]:

min_val = []
max_val = []
mean_diff = []
question = []
mean_dia = []
mean = []
country = []
z = 1.645
for j in countries:
    for i in names:
        n1 = len(df[(df["Diabetes"]==1) & (df["Country"]==j)])
        n2 = len(df[df["Country"]==j])
        x1 = np.mean(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        x2 = np.mean(df[i][df["Country"]==j])
        s1 = np.var(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        s2 = np.var(df[i][df["Country"]==j])
        std_er = math.sqrt((s1/n1)+(s2/n2))
        r1 = abs(x1-x2) + (z * (std_er))
        r2 = abs(x1-x2) - (z * (std_er))
        min_val.append(r1)
        max_val.append(r2)
        mean_diff.append(abs(x1-x2))
        question.append(i)
        mean_dia.append(x1)
        mean.append(x2)
        country.append(j)
result2 = pd.DataFrame({"Country":country,"Question":question,"Mean_diff":mean_diff,"Max":max_val,"Min":min_val,"Mean_dia":mean_dia,"Mean_Pop":mean})    


# In[113]:

result2["Pop_max"] = result2["Mean_Pop"] + result2["Max"]
result2["Pop_min"] = result2["Mean_Pop"] + result2["Min"]


# In[114]:

min_val = []
max_val = []
mean_diff = []
question = []
mean_dia = []
mean = []
country = []
z = 1.44
for j in countries:
    for i in names:
        n1 = len(df[(df["Diabetes"]==1) & (df["Country"]==j)])
        n2 = len(df[df["Country"]==j])
        x1 = np.mean(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        x2 = np.mean(df[i][df["Country"]==j])
        s1 = np.var(df[i][(df["Diabetes"]==1) & (df["Country"]==j)])
        s2 = np.var(df[i][df["Country"]==j])
        std_er = math.sqrt((s1/n1)+(s2/n2))
        r1 = abs(x1-x2) + (z * (std_er))
        r2 = abs(x1-x2) - (z * (std_er))
        min_val.append(r1)
        max_val.append(r2)
        mean_diff.append(abs(x1-x2))
        question.append(i)
        mean_dia.append(x1)
        mean.append(x2)
        country.append(j)
result3 = pd.DataFrame({"Country":country,"Question":question,"Mean_diff":mean_diff,"Max":max_val,"Min":min_val,"Mean_dia":mean_dia,"Mean_Pop":mean})    


# In[115]:

result3["Pop_max"] = result3["Mean_Pop"] + result3["Max"]
result3["Pop_min"] = result3["Mean_Pop"] + result3["Min"]


# In[116]:

path = "C:/Users/css112831/Desktop/Google - Sanofi (Data Sources)/data/Ztest.xlsx"
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')


# In[117]:


result.to_excel(writer,
                sheet_name='Sheet1',index = False, engine='openpyxl')
result1.to_excel(writer,
                sheet_name='Sheet2',index = False, engine='openpyxl')
result2.to_excel(writer,
                sheet_name='Sheet3',index = False, engine='openpyxl')
result3.to_excel(writer,
                sheet_name='Sheet4',index = False, engine='openpyxl')
writer.save()
writer.close()


# In[ ]:



