
# coding: utf-8

# In[1]:

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


# In[77]:

df = pd.read_csv("..\input_files\simulation_new_theme_5.csv")

df_new=df.drop(['Cluster'],axis=1)


# In[13]:

countries = list(df_new['Country'].unique())
themes = ['Barriers','Cues to action','Doctor relationship','Information on Flu','Self efficacy','Threat perception','Vaccination attitude']
best_cluster_theme = pd.DataFrame()
rows = 0
for c in countries:
    clusters_c = list(df.loc[(df['Country'] == c),'Cluster'].unique())
    for cl in clusters_c:
        questions = ['Flu jab affordability','Flu jab accessibility','Time availability for flu jab','Informed to get the flu jab',
                     'Relatives recommends flu jab','Relatives help in health decisions','Doctor recommends flu jab',
                     'Doctor involvement in health decisions','Trust in doctor about flu jab info','Knowledge of flu',
                     'Trust info provided by social media','Trust the info provided by Health Authorities','Trust the info provided by TV/radio/newspapers',
                     'Know enough about the flu jab','Self health rating','Taking action to protect against flu',
                     'Flu could make me severely ill','Perception of sick time from flu','Scared of getting the flu',
                     'Favouring vaccination','Not scared of getting a flu jab','Open to flu jabs']
        for q in questions:
            df_sub = df[(df.Country == c) & (df.Cluster == cl) & (df.Adherence_Score_2 == 0)]
            avg = df_sub[q].mean()
            std_ = df_sub[q].std()
            best_cluster_theme.loc[rows,'Country'] = c
            best_cluster_theme.loc[rows,'Cluster'] = cl
            if (avg+(std_) > 9.5):
                best_cluster_theme.loc[rows,q] = 9
            else:
                best_cluster_theme.loc[rows,q] = avg+(std_)
        rows = rows+1
            
                
                


# In[78]:

best_cluster_theme


# In[121]:

best_cluster_theme.to_csv("C:\Users\css113429\Desktop\Google Sanofi\\best_cluster_theme.csv")


# In[79]:

df_new = df.drop(['Intent to vaccinate','Affordability','Accessibility','Time availability'],axis=1)

df_new.head()
countries = list(df_new['Country'].unique())
age_bucket = list(df_new["Age bucket"].unique())
diabetes_ = list(df_new["Diabetes"].unique())
chronic_conditions = list(df_new["Chronic conditions"].unique())
clusters = list(df_new["Cluster"].unique())

df_new2 = df_new.drop(['Cluster','Adherence_Score_2','Barriers_score','Cues to action_score','Doctor relationship_score',
                       'Information on Flu_score','Self efficacy_score','Threat perception_score','Vaccination attitude_score'],axis=1)

df_dummied = pd.get_dummies(df_new2)

#Country wise
for i in countries:
    exec('df_country = df_dumssmied[df_dummied.Country_{} == 1]'.format(i))
    data_x_c = df_country.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country[['Adherence_Score']]
    exec('clf_{} = svm.SVC(kernel="linear")'.format(i))
    exec('clf_{}.fit(data_x_c, data_y_c)'.format(i))
    exec('imp_{}= pd.DataFrame(list(data_x_c.columns))'.format(i))
    exec('imp_{}["coefficient"]=clf_{}.coef_[0]'.format(i,i))
    exec('svm_weights_selected = (clf_{}.coef_ ** 2).sum(axis=0)'.format(i))
    svm_weights_selected /= svm_weights_selected.max()
    exec('imp_{}["importance"]=svm_weights_selected'.format(i))
    exec('imp_{}.columns=["Column_name","Coefficient","Importance"]'.format(i))
    exec('imp_{}.to_csv("C:\Users\css113429\Desktop\Google Sanofi\\SVM_importance_latest_{}.csv")'.format(i,i))


data_x = df_dummied.drop(['Adherence_Score'],axis=1)
data_y = df_dummied[['Adherence_Score']]
clf_Overall = svm.SVC(kernel="linear")
clf_Overall.fit(data_x, data_y)
overall_imp= pd.DataFrame(list(data_x.columns))
overall_imp["coefficient"]=clf_Overall.coef_[0]
svm_weights_selected = (clf_Overall.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()
overall_imp['overall_importance']=svm_weights_selected
overall_imp.columns=['Column_name','Coefficient','Importance']
overall_imp.to_csv("C:\Users\css113429\Desktop\Google Sanofi\\SVM_importance_latest_Overall.csv")


# In[46]:

def adherence_calc(country, cluster,theme):
    df_c = df_new[df_new.Country == country]
    best_cluster_theme_new = best_cluster_theme[best_cluster_theme.Country == country]
    df_c_2 = df_c.copy()
    df_c = df_c[(df_c['Cluster'] == cluster)]
    df_c_3 = df_c.copy()
    best_cluster_theme_new = best_cluster_theme_new[best_cluster_theme_new.Cluster == cluster]


    barriers = ['Flu jab affordability','Flu jab accessibility','Time availability for flu jab']
    cues = ['Informed to get the flu jab','Relatives recommends flu jab','Relatives help in health decisions']
    doc_rel = ['Doctor recommends flu jab','Doctor involvement in health decisions','Trust in doctor about flu jab info']
    flu_know = ['Knowledge of flu','Trust info provided by social media','Trust the info provided by Health Authorities','Trust the info provided by TV/radio/newspapers']
    threat = ['Flu could make me severely ill','Perception of sick time from flu','Scared of getting the flu']
    vac = ['Favouring vaccination','Not scared of getting a flu jab','Open to flu jabs']
    self = ['Know enough about the flu jab','Self health rating','Taking action to protect against flu']
    
    if (theme == 'barriers'):
        y = barriers
    elif (theme == 'cues'):
        y = cues
    elif (theme == 'doc_rel'):
        y = doc_rel
    elif (theme == 'flu_know'):
        y = flu_know
    elif (theme == 'threat'):
        y = threat
    elif (theme == 'self'):
        y = self
    elif (theme == 'vac'):
        y = vac
    
    value_list = [0]*len(y)
    
    for i in range(len(y)):
        value_list[i] = float(best_cluster_theme_new.loc[:,y[i]])
        
    no_change = 0
    
    for c in range(len(value_list)):
        if (min(df_c.loc[(df_c['Adherence_Score_2'] == 0),y[c]]) < value_list[c]):
            df_c.loc[(df_c[y[c]] < value_list[c]) & (df_c['Adherence_Score_2'] == 0),y[c]] = value_list[c]
            no_change = 1
    
    calc_theme_score = df_c[y].mean(axis=1).mean()
    act_theme_score = df_c_3[y].mean(axis=1).mean()
    
    df_c = df_c.drop(['Cluster','Adherence_Score_2','Barriers_score','Cues to action_score','Doctor relationship_score',
                       'Information on Flu_score','Self efficacy_score','Threat perception_score','Vaccination attitude_score'],axis=1)

    dummy_data = pd.get_dummies(df_c)

    columns_req = list(df_dummied.columns)

    for i in columns_req:
        if i not in list(dummy_data.columns):
            dummy_data[i] = 0


    exec('df_country = dummy_data[dummy_data.Country_{} == 1]'.format(country))
    df_country2 = df_country[df_country.Adherence_Score < 80]
    data_x_c = df_country2.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country2[['Adherence_Score']]

    exec('y_pred = pd.DataFrame(clf_{}.predict(data_x_c))'.format(country))
    diff_cluster_sum = float(sum(df_c_2.loc[df_c_2['Cluster'] != cluster,'Adherence_Score']))
    cluster_sum = float(sum(y_pred[0]))+float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score']))
    change_rows = float(len(y_pred[0]))
    non_change_rows = len(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])
    cluster_rows = float(len(df_country.loc[:,'Adherence_Score']))
    total_rows = float(len(df_c_2.loc[:,'Adherence_Score']))
    mean_adh = float(((diff_cluster_sum + cluster_sum)/total_rows))
    clust_adherence = (float(sum(y_pred[0])) + float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])))/float(len(df_country.loc[:,'Adherence_Score']))
    act_clust_adherence = float(df_c[['Adherence_Score']].mean())
    con_adherence = mean_adh
    act_con_adherence = float(df_c_2[['Adherence_Score']].mean())
    #print('{}'.format(country))
    #print('{}'.format(cluster))
    #print('{}'.format(theme))
    #print('{0:0.2f}%'.format(mean_adh))
    return country,cluster,theme,calc_theme_score,act_theme_score,mean_adh,clust_adherence,act_con_adherence,act_clust_adherence


# In[70]:

value_list


# In[53]:

adherence_calc('US','US2','barriers')


# In[54]:

themes = ['barriers','cues','doc_rel','flu_know','threat','self','vac']
recorded = pd.DataFrame()
rows = 0
for c in countries:
    clusters_c = list(df.loc[(df['Country'] == c),'Cluster'].unique())
    for cl in clusters_c:
        for th in themes:
            country,cluster,theme,calc_theme_score,act_theme_score,mean_adh,clust_adher,act_mean_adh,act_clust_adher  = adherence_calc(c,cl,th)
            recorded.loc[rows,'Country'] = country
            recorded.loc[rows,'Cluster'] = cluster
            recorded.loc[rows,'Theme'] = theme
            recorded.loc[rows,'CalcThemeScore'] = calc_theme_score
            recorded.loc[rows,'ActualThemeScore'] = act_theme_score
            recorded.loc[rows,'CalcCountryAdherence'] = mean_adh
            recorded.loc[rows,'CalcClusterAdherence'] = clust_adher
            recorded.loc[rows,'ActualCountryAdherence'] = act_mean_adh
            recorded.loc[rows,'ActualClusterAdherence'] = act_clust_adher
            rows = rows+1


# In[228]:

recorded


# In[229]:

recorded.to_csv("C:\Users\css113429\Desktop\Google Sanofi\\recorded_everything9.csv")


# In[85]:

def adherence_calc_2themes(country, cluster,theme1, theme2):
    df_c = df_new[df_new.Country == country]
    best_cluster_theme_new = best_cluster_theme[best_cluster_theme.Country == country]
    df_c_2 = df_c.copy()
    df_c = df_c[(df_c['Cluster'] == cluster)]
    df_c_3 = df_c.copy()
    best_cluster_theme_new = best_cluster_theme_new[best_cluster_theme_new.Cluster == cluster]


    barriers = ['Flu jab affordability','Flu jab accessibility','Time availability for flu jab']
    cues = ['Informed to get the flu jab','Relatives recommends flu jab','Relatives help in health decisions']
    doc_rel = ['Doctor recommends flu jab','Doctor involvement in health decisions','Trust in doctor about flu jab info']
    flu_know = ['Knowledge of flu','Trust info provided by social media','Trust the info provided by Health Authorities','Trust the info provided by TV/radio/newspapers']
    threat = ['Flu could make me severely ill','Perception of sick time from flu','Scared of getting the flu']
    vac = ['Favouring vaccination','Not scared of getting a flu jab','Open to flu jabs']
    self = ['Know enough about the flu jab','Self health rating','Taking action to protect against flu']
    
    if (theme1 == 'barriers'):
        y1 = barriers
    elif (theme1 == 'cues'):
        y1 = cues
    elif (theme1 == 'doc_rel'):
        y1 = doc_rel
    elif (theme1 == 'flu_know'):
        y1 = flu_know
    elif (theme1 == 'threat'):
        y1 = threat
    elif (theme1 == 'self'):
        y1 = self
    elif (theme1 == 'vac'):
        y1 = vac
    
    
    if (theme2 == 'barriers'):
        y2 = barriers
    elif (theme2 == 'cues'):
        y2 = cues
    elif (theme2 == 'doc_rel'):
        y2 = doc_rel
    elif (theme2 == 'flu_know'):
        y2 = flu_know
    elif (theme2 == 'threat'):
        y2 = threat
    elif (theme2 == 'self'):
        y2 = self
    elif (theme2 == 'vac'):
        y2 = vac
    
    value_list1 = [0]*len(y1)
    value_list2 = [0]*len(y2)
    
    for i in range(len(y1)):
        value_list1[i] = float(best_cluster_theme_new.loc[:,y1[i]])
    
    for i in range(len(y2)):
        value_list2[i] = float(best_cluster_theme_new.loc[:,y2[i]])
        
    no_change = 0
    
    for c in range(len(value_list1)):
        if (min(df_c.loc[(df_c['Adherence_Score_2'] == 0),y1[c]]) < value_list1[c]):
            df_c.loc[(df_c[y1[c]] < value_list1[c]) & (df_c['Adherence_Score_2'] == 0),y1[c]] = value_list1[c]
            no_change = 1
            
    for c in range(len(value_list2)):
        if (min(df_c.loc[(df_c['Adherence_Score_2'] == 0),y2[c]]) < value_list2[c]):
            df_c.loc[(df_c[y2[c]] < value_list2[c]) & (df_c['Adherence_Score_2'] == 0),y2[c]] = value_list2[c]
            no_change = 1
    
    calc_theme_score1 = df_c[y1].mean(axis=1).mean()
    act_theme_score1 = df_c_3[y1].mean(axis=1).mean()
    
    calc_theme_score2 = df_c[y2].mean(axis=1).mean()
    act_theme_score2 = df_c_3[y2].mean(axis=1).mean()
    
    df_c = df_c.drop(['Cluster','Adherence_Score_2','Barriers_score','Cues to action_score','Doctor relationship_score',
                       'Information on Flu_score','Self efficacy_score','Threat perception_score','Vaccination attitude_score'],axis=1)

    dummy_data = pd.get_dummies(df_c)

    columns_req = list(df_dummied.columns)

    for i in columns_req:
        if i not in list(dummy_data.columns):
            dummy_data[i] = 0


    exec('df_country = dummy_data[dummy_data.Country_{} == 1]'.format(country))
    df_country2 = df_country[df_country.Adherence_Score < 80]
    data_x_c = df_country2.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country2[['Adherence_Score']]

    exec('y_pred = pd.DataFrame(clf_{}.predict(data_x_c))'.format(country))
    diff_cluster_sum = float(sum(df_c_2.loc[df_c_2['Cluster'] != cluster,'Adherence_Score']))
    cluster_sum = float(sum(y_pred[0]))+float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score']))
    change_rows = float(len(y_pred[0]))
    non_change_rows = len(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])
    cluster_rows = float(len(df_country.loc[:,'Adherence_Score']))
    total_rows = float(len(df_c_2.loc[:,'Adherence_Score']))
    mean_adh = float(((diff_cluster_sum + cluster_sum)/total_rows))
    clust_adherence = (float(sum(y_pred[0])) + float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])))/float(len(df_country.loc[:,'Adherence_Score']))
    act_clust_adherence = float(df_c[['Adherence_Score']].mean())
    con_adherence = mean_adh
    act_con_adherence = float(df_c_2[['Adherence_Score']].mean())
    #print('{}'.format(country))
    #print('{}'.format(cluster))
    #print('{}'.format(theme))
    #print('{0:0.2f}%'.format(mean_adh))
    return country,cluster,theme1,theme2,calc_theme_score1,act_theme_score1,calc_theme_score2,act_theme_score2,mean_adh,clust_adherence,act_con_adherence,act_clust_adherence


# In[87]:

adherence_calc_2themes('US','US1','threat','cues')


# In[88]:

themes = ['barriers','cues','doc_rel','flu_know','threat','self','vac']
recorded_2themes = pd.DataFrame()
rows = 0
for c in countries:
    clusters_c = list(df.loc[(df['Country'] == c),'Cluster'].unique())
    for cl in clusters_c:
        for th1 in themes:
            for th2 in themes:
                country,cluster,theme1,theme2,calc_theme_score1,act_theme_score1,calc_theme_score2,act_theme_score2,mean_adh,clust_adher,act_mean_adh,act_clust_adher  = adherence_calc_2themes(c,cl,th1,th2)
                recorded_2themes.loc[rows,'Country'] = country
                recorded_2themes.loc[rows,'Cluster'] = cluster
                recorded_2themes.loc[rows,'Theme1'] = theme1
                recorded_2themes.loc[rows,'Theme2'] = theme2
                recorded_2themes.loc[rows,'CalcThemeScore1'] = calc_theme_score1
                recorded_2themes.loc[rows,'ActualThemeScore1'] = act_theme_score1
                recorded_2themes.loc[rows,'CalcThemeScore2'] = calc_theme_score2
                recorded_2themes.loc[rows,'ActualThemeScore2'] = act_theme_score2
                recorded_2themes.loc[rows,'CalcCountryAdherence'] = mean_adh
                recorded_2themes.loc[rows,'CalcClusterAdherence'] = clust_adher
                recorded_2themes.loc[rows,'ActualCountryAdherence'] = act_mean_adh
                recorded_2themes.loc[rows,'ActualClusterAdherence'] = act_clust_adher
                rows = rows+1


# In[89]:

recorded_2themes.to_csv("C:\Users\css113429\Desktop\Google Sanofi\\recorded_2themes_v2.csv")


# In[90]:

countries = list(df_new['Country'].unique())
themes = ['Barriers','Cues to action','Doctor relationship','Information on Flu','Self efficacy','Threat perception','Vaccination attitude']
best_country_theme = pd.DataFrame()
rows = 0
for c in countries:
    questions = ['Flu jab affordability','Flu jab accessibility','Time availability for flu jab','Informed to get the flu jab',
                 'Relatives recommends flu jab','Relatives help in health decisions','Doctor recommends flu jab',
                 'Doctor involvement in health decisions','Trust in doctor about flu jab info','Knowledge of flu',
                 'Trust info provided by social media','Trust the info provided by Health Authorities','Trust the info provided by TV/radio/newspapers',
                 'Know enough about the flu jab','Self health rating','Taking action to protect against flu',
                 'Flu could make me severely ill','Perception of sick time from flu','Scared of getting the flu',
                 'Favouring vaccination','Not scared of getting a flu jab','Open to flu jabs']
    for q in questions:
        df_sub = df[(df.Country == c) & (df.Adherence_Score_2 == 0)]
        avg = df_sub[q].mean()
        std_ = df_sub[q].std()
        best_country_theme.loc[rows,'Country'] = c
        if (avg+(std_) > 9.5):
            best_country_theme.loc[rows,q] = 9
        else:
            best_country_theme.loc[rows,q] = avg+(std_)
    rows = rows+1


# In[91]:

def adherence_calc_2themes_country(country, theme1, theme2):
    df_c = df_new[df_new.Country == country]
    best_cluster_theme_new = best_country_theme[best_country_theme.Country == country]
    df_c_2 = df_c.copy()
    df_c_3 = df_c.copy()
    cluster = 'All'


    barriers = ['Flu jab affordability','Flu jab accessibility','Time availability for flu jab']
    cues = ['Informed to get the flu jab','Relatives recommends flu jab','Relatives help in health decisions']
    doc_rel = ['Doctor recommends flu jab','Doctor involvement in health decisions','Trust in doctor about flu jab info']
    flu_know = ['Knowledge of flu','Trust info provided by social media','Trust the info provided by Health Authorities','Trust the info provided by TV/radio/newspapers']
    threat = ['Flu could make me severely ill','Perception of sick time from flu','Scared of getting the flu']
    vac = ['Favouring vaccination','Not scared of getting a flu jab','Open to flu jabs']
    self = ['Know enough about the flu jab','Self health rating','Taking action to protect against flu']
    
    if (theme1 == 'barriers'):
        y1 = barriers
    elif (theme1 == 'cues'):
        y1 = cues
    elif (theme1 == 'doc_rel'):
        y1 = doc_rel
    elif (theme1 == 'flu_know'):
        y1 = flu_know
    elif (theme1 == 'threat'):
        y1 = threat
    elif (theme1 == 'self'):
        y1 = self
    elif (theme1 == 'vac'):
        y1 = vac
    
    
    if (theme2 == 'barriers'):
        y2 = barriers
    elif (theme2 == 'cues'):
        y2 = cues
    elif (theme2 == 'doc_rel'):
        y2 = doc_rel
    elif (theme2 == 'flu_know'):
        y2 = flu_know
    elif (theme2 == 'threat'):
        y2 = threat
    elif (theme2 == 'self'):
        y2 = self
    elif (theme2 == 'vac'):
        y2 = vac
    
    value_list1 = [0]*len(y1)
    value_list2 = [0]*len(y2)
    
    for i in range(len(y1)):
        value_list1[i] = float(best_cluster_theme_new.loc[:,y1[i]])
    
    for i in range(len(y2)):
        value_list2[i] = float(best_cluster_theme_new.loc[:,y2[i]])
        
    no_change = 0
    
    for c in range(len(value_list1)):
        if (min(df_c.loc[(df_c['Adherence_Score_2'] == 0),y1[c]]) < value_list1[c]):
            df_c.loc[(df_c[y1[c]] < value_list1[c]) & (df_c['Adherence_Score_2'] == 0),y1[c]] = value_list1[c]
            no_change = 1
            
    for c in range(len(value_list2)):
        if (min(df_c.loc[(df_c['Adherence_Score_2'] == 0),y2[c]]) < value_list2[c]):
            df_c.loc[(df_c[y2[c]] < value_list2[c]) & (df_c['Adherence_Score_2'] == 0),y2[c]] = value_list2[c]
            no_change = 1
    
    calc_theme_score1 = df_c[y1].mean(axis=1).mean()
    act_theme_score1 = df_c_3[y1].mean(axis=1).mean()
    
    calc_theme_score2 = df_c[y2].mean(axis=1).mean()
    act_theme_score2 = df_c_3[y2].mean(axis=1).mean()
    
    df_c = df_c.drop(['Cluster','Adherence_Score_2','Barriers_score','Cues to action_score','Doctor relationship_score',
                       'Information on Flu_score','Self efficacy_score','Threat perception_score','Vaccination attitude_score'],axis=1)

    dummy_data = pd.get_dummies(df_c)

    columns_req = list(df_dummied.columns)

    for i in columns_req:
        if i not in list(dummy_data.columns):
            dummy_data[i] = 0


    exec('df_country = dummy_data[dummy_data.Country_{} == 1]'.format(country))
    df_country2 = df_country[df_country.Adherence_Score < 80]
    data_x_c = df_country2.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country2[['Adherence_Score']]

    exec('y_pred = pd.DataFrame(clf_{}.predict(data_x_c))'.format(country))
    diff_cluster_sum = 0
    cluster_sum = float(sum(y_pred[0]))+float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score']))
    change_rows = float(len(y_pred[0]))
    non_change_rows = len(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])
    cluster_rows = float(len(df_country.loc[:,'Adherence_Score']))
    total_rows = float(len(df_c_2.loc[:,'Adherence_Score']))
    mean_adh = float(((diff_cluster_sum + cluster_sum)/total_rows))
    clust_adherence = (float(sum(y_pred[0])) + float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])))/float(len(df_country.loc[:,'Adherence_Score']))
    act_clust_adherence = float(df_c[['Adherence_Score']].mean())
    con_adherence = mean_adh
    act_con_adherence = float(df_c_2[['Adherence_Score']].mean())
    #print('{}'.format(country))
    #print('{}'.format(cluster))
    #print('{}'.format(theme))
    #print('{0:0.2f}%'.format(mean_adh))
    return country,cluster,theme1,theme2,calc_theme_score1,act_theme_score1,calc_theme_score2,act_theme_score2,mean_adh,clust_adherence,act_con_adherence,act_clust_adherence


# In[95]:

adherence_calc_2themes_country('FR', 'doc_rel','flu_know')


# In[96]:

themes = ['barriers','cues','doc_rel','flu_know','threat','self','vac']
recorded_2themes_country = pd.DataFrame()
rows = 0
for c in countries:
    for th1 in themes:
        for th2 in themes:
            country,cluster,theme1,theme2,calc_theme_score1,act_theme_score1,calc_theme_score2,act_theme_score2,mean_adh,clust_adher,act_mean_adh,act_clust_adher  = adherence_calc_2themes_country(c,th1,th2)
            recorded_2themes_country.loc[rows,'Country'] = country
            recorded_2themes_country.loc[rows,'Cluster'] = cluster
            recorded_2themes_country.loc[rows,'Theme1'] = theme1
            recorded_2themes_country.loc[rows,'Theme2'] = theme2
            recorded_2themes_country.loc[rows,'CalcThemeScore1'] = calc_theme_score1
            recorded_2themes_country.loc[rows,'ActualThemeScore1'] = act_theme_score1
            recorded_2themes_country.loc[rows,'CalcThemeScore2'] = calc_theme_score2
            recorded_2themes_country.loc[rows,'ActualThemeScore2'] = act_theme_score2
            recorded_2themes_country.loc[rows,'CalcCountryAdherence'] = mean_adh
            recorded_2themes_country.loc[rows,'CalcClusterAdherence'] = clust_adher
            recorded_2themes_country.loc[rows,'ActualCountryAdherence'] = act_mean_adh
            recorded_2themes_country.loc[rows,'ActualClusterAdherence'] = act_clust_adher
            rows = rows+1


# In[97]:

recorded_2themes_country.to_csv("..\input_files\\recorded_2themes_countrylevel_v2.csv")

