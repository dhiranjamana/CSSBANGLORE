
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

df = pd.read_csv("C:\Users\css113429\Desktop\Google Sanofi\\Combined_data_cleaned_v4_drilldown.csv")


# In[ ]:

df_new = df.drop(['Adherence_Score','DOB','Monthly household income','Flu jab last 6 months','Flu jab frequency','Age',
            'State (District for China)','City','Zip codes (For UK,US and FR)','Likeliness to get flu jab coming season'],axis=1)

df_new.head()
countries = list(df_new['Country'].unique())

df_new_lt65 = df_new[df_new['Age bucket'] != '65+']
df_new_65plus = df_new[df_new['Age bucket'] == '65+']
df_new_65plus = df_new_65plus.drop(['Age bucket'],axis=1)

df_dummied_lt65 = pd.get_dummies(df_new_lt65)
df_dummied_65plus = pd.get_dummies(df_new_65plus)
df_dummied_overall = pd.get_dummies(df_new)


# In[ ]:

#Country wise 65+
for i in countries:
    exec('df_country = df_dummied_65plus[df_dummied_65plus.Country_{} == 1]'.format(i))
    data_x_c = df_country.drop(['Adherence_Score_2'],axis=1)
    data_y_c = df_country[['Adherence_Score_2']]
    exec('clf_{} = svm.SVC(kernel="linear")'.format(i))
    exec('clf_{}.fit(data_x_c, data_y_c)'.format(i))
    exec('imp_{}= pd.DataFrame(list(data_x_c.columns))'.format(i))
    exec('imp_{}["coefficient"]=clf_{}.coef_[0]'.format(i,i))
    exec('svm_weights_selected = (clf_{}.coef_ ** 2).sum(axis=0)'.format(i))
    svm_weights_selected /= svm_weights_selected.max()
    exec('imp_{}["importance"]=svm_weights_selected'.format(i))
    exec('imp_{}.columns=["Column_name","Coefficient","Importance"]'.format(i))

