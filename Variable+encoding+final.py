
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
import random
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import cufflinks as cf
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.stats import ttest_ind,f_oneway
import dash_auth
import os
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')


# In[42]:

df = pd.read_csv("..\input_files\\Combined_data_cleaned_v4_drilldown.csv")
variable_encoding = pd.read_csv("..\input_files\\variable_encoding_2_65plus.csv")
pam_mapping = pd.read_csv("..\input_files\\PAM_ScoreQuestions_65plus_2.csv")


# In[43]:

themes = ['External Barriers','Doctor relation','Cues to Action','Knowledge on flu','Self efficacy','Vulnerability','Vaccination attitude','Vulnerability_old','Vaccination attitude_old']


# In[44]:

#Encoding variables for SVM model and simulation
for i in list(variable_encoding["Column"].unique()):
    data_to_merge = variable_encoding.loc[variable_encoding["Column"]==i,["Column_Value","Encoding"]]
    if (df[i].dtype != np.object):
        data_to_merge["Column_Value"] = data_to_merge["Column_Value"].astype(int)
    exec('data_to_merge.columns = ["{}","{}_encoded"]'.format(i,i))
    df = pd.merge(df,data_to_merge, on=[i], how="left")
    exec('df.drop(["{}"], axis=1)'.format(i))


# In[45]:

#Adding theme scores to the data
df_output = df.copy()
for i in list(df_output['Country'].unique()):
    for j in themes:
        theme_questions = list(pam_mapping.loc[pam_mapping['Theme'] == j,i])
        df_output.loc[df_output['Country'] == i,j] = df_output.loc[df_output['Country'] == i,theme_questions].mean(axis=1)

