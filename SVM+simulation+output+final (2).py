
# coding: utf-8

# In[1]:

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


# In[22]:

df = pd.read_csv("..\input_files\\encoded_data_vaxitrend_new_v2.csv")
pam_mapping = pd.read_csv("..\input_files\\PAM_ScoreQuestions_65plus_2.csv")


# In[23]:

df.loc[df['Adherence_Score'] >= 80,'Adherence_Score_2'] = 1
df.loc[df['Adherence_Score'] < 80,'Adherence_Score_2'] = 0


# In[7]:

countries = list(df['Country'].unique())
best_cluster_theme = pd.DataFrame()
rows = 0
for c in countries:
    clusters_c = list(df.loc[(df['Country'] == c),'Cluster'].unique())
    for cl in clusters_c:
        questions = ['Can afford flu shot_encoded','Confident I can get a flu jab if I want one_encoded',
                     'I can make time to get the flu jab_encoded','Informed about whether I should or could get the flu jab_encoded',
                     'My relatives or close friends thinks I should get a flu jab_encoded','Parents / spouse / relatives / friends help me make decisions about my health_encoded',
                     'My doctor thinks I should get a flu jab_encoded','Participation with doctor_encoded',
                     'Trust the information provided by my doctor about the flu jab_encoded','Knowledge of flu_encoded',
                     'Trust information provided by general websites, YouTube, blogs, Facebook or Twitter about the flu jab_encoded',
                     'Trust the information provided by Health Authorities_encoded',
                     'Trust the information provided by news reports on TV & radio or newspapers about the flu jab_encoded',
                     'Know enough about the flu jab to make an informed decision on vaccination_encoded','self health rating_encoded',
                     'Taking action to try to protect myself against the flu_encoded','Flu could make me severely ill_encoded',
                     'I believe that if I got the flu I would have to stay in bed for_encoded','Scared of getting the flu_encoded',
                     'Favouring vaccination_encoded','I am scared of getting a flu jab_encoded','Open to the idea of receiving the flu jab, in principle_encoded',
                     'If I got the flu I would feel sicker than other people my age_encoded',
                     'Worried about passing flu to other people if no flu jab_encoded','Feel it is important that I get vaccinated_encoded',
                     'Would regret not getting the jab_encoded',
                     'If I dont get a flu jab this autumn/winter, the chances of me getting flu are_encoded',
                     'Confident I can avoid getting the flu, even without the flu jab_encoded',
                     'Scared to pass flu to a vulnerable relative_encoded','If I got the flu I worry that I would miss out on important activities or events_encoded',
                     'With no flu jab I would feel very vulnerable to the flu_encoded','Vaccination forms part of a healthy lifestyle_encoded']

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


# In[25]:

subset = ['Country','Cluster','Age bucket','Age','Income bucket','Diabetes','Chronic conditions','Adherence_Score','Adherence_Score_2','Can afford flu shot_encoded','Confident I can get a flu jab if I want one_encoded','I can make time to get the flu jab_encoded','Informed about whether I should or could get the flu jab_encoded','My relatives or close friends thinks I should get a flu jab_encoded','Parents / spouse / relatives / friends help me make decisions about my health_encoded','My doctor thinks I should get a flu jab_encoded','Participation with doctor_encoded','Trust the information provided by my doctor about the flu jab_encoded','Knowledge of flu_encoded','Trust information provided by general websites, YouTube, blogs, Facebook or Twitter about the flu jab_encoded','Trust the information provided by Health Authorities_encoded','Trust the information provided by news reports on TV & radio or newspapers about the flu jab_encoded','Know enough about the flu jab to make an informed decision on vaccination_encoded','self health rating_encoded','Taking action to try to protect myself against the flu_encoded','Flu could make me severely ill_encoded','I believe that if I got the flu I would have to stay in bed for_encoded','Scared of getting the flu_encoded','Favouring vaccination_encoded','I am scared of getting a flu jab_encoded','Open to the idea of receiving the flu jab, in principle_encoded','If I got the flu I would feel sicker than other people my age_encoded','Worried about passing flu to other people if no flu jab_encoded','Feel it is important that I get vaccinated_encoded','Would regret not getting the jab_encoded','If I dont get a flu jab this autumn/winter, the chances of me getting flu are_encoded','Confident I can avoid getting the flu, even without the flu jab_encoded','Scared to pass flu to a vulnerable relative_encoded','If I got the flu I worry that I would miss out on important activities or events_encoded','With no flu jab I would feel very vulnerable to the flu_encoded','Vaccination forms part of a healthy lifestyle_encoded']
df_new = df[subset]
df_new2 = df_new.drop(['Cluster'],axis=1)


# In[26]:

df_dummied = pd.get_dummies(df_new2)

#Country wise
for i in countries:
    exec('df_country = df_dummied[df_dummied.Country_{} == 1]'.format(i))
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


# In[27]:

def adherence_calc_2themes(country, cluster,theme1, theme2):
    df_c = df_new[df_new.Country == country]
    best_cluster_theme_new = best_cluster_theme[best_cluster_theme.Country == country]
    df_c_2 = df_c.copy()
    df_c = df_c[(df_c['Cluster'] == cluster)]
    df_c_3 = df_c.copy()
    best_cluster_theme_new = best_cluster_theme_new[best_cluster_theme_new.Cluster == cluster]


    y1 = list(pam_mapping.loc[pam_mapping['Theme'] == theme1,country])
    y2 = list(pam_mapping.loc[pam_mapping['Theme'] == theme2,country])
    
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
    
    df_c = df_c.drop(['Cluster','Adherence_Score_2'],axis=1)

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


# In[30]:

themes = ['External Barriers','Doctor relation','Cues to Action','Knowledge on flu','Self efficacy','Vulnerability_old','Vaccination attitude_old']
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
                if (theme1 == 'Vulnerability_old'):
                    theme1 = 'Vulnerability'
                elif (theme1 == 'Vaccination attitude_old'):
                    theme1 = 'Vaccination attitude'
                if (theme2 == 'Vulnerability_old'):
                    theme2 = 'Vulnerability'
                elif (theme2 == 'Vaccination attitude_old'):
                    theme2 = 'Vaccination attitude'
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
                if(mean_adh > act_mean_adh):
                    finalcon_adh = mean_adh
                    finalclust_adh = clust_adher
                else:
                    finalcon_adh = act_mean_adh
                    finalclust_adh = act_clust_adher
                recorded_2themes.loc[rows,'FinalCountryAdherence'] = finalcon_adh
                recorded_2themes.loc[rows,'FinalClusterAdherence'] = finalclust_adh
                recorded_2themes.loc[rows,'Country_diff'] = float(finalcon_adh - act_mean_adh)
                recorded_2themes.loc[rows,'Cluster_diff'] = float(finalclust_adh - act_clust_adher)
                rows = rows+1


# ##65+

# In[31]:

countries = list(df_new['Country'].unique())
best_country_theme = pd.DataFrame()
rows = 0
for c in countries:
    questions = ['Can afford flu shot_encoded','Confident I can get a flu jab if I want one_encoded',
                     'I can make time to get the flu jab_encoded','Informed about whether I should or could get the flu jab_encoded',
                     'My relatives or close friends thinks I should get a flu jab_encoded','Parents / spouse / relatives / friends help me make decisions about my health_encoded',
                     'My doctor thinks I should get a flu jab_encoded','Participation with doctor_encoded',
                     'Trust the information provided by my doctor about the flu jab_encoded','Knowledge of flu_encoded',
                     'Trust information provided by general websites, YouTube, blogs, Facebook or Twitter about the flu jab_encoded',
                     'Trust the information provided by Health Authorities_encoded',
                     'Trust the information provided by news reports on TV & radio or newspapers about the flu jab_encoded',
                     'Know enough about the flu jab to make an informed decision on vaccination_encoded','self health rating_encoded',
                     'Taking action to try to protect myself against the flu_encoded','Flu could make me severely ill_encoded',
                     'I believe that if I got the flu I would have to stay in bed for_encoded','Scared of getting the flu_encoded',
                     'Favouring vaccination_encoded','I am scared of getting a flu jab_encoded','Open to the idea of receiving the flu jab, in principle_encoded',
                     'If I got the flu I would feel sicker than other people my age_encoded',
                     'Worried about passing flu to other people if no flu jab_encoded','Feel it is important that I get vaccinated_encoded',
                     'Would regret not getting the jab_encoded',
                     'If I dont get a flu jab this autumn/winter, the chances of me getting flu are_encoded',
                     'Confident I can avoid getting the flu, even without the flu jab_encoded',
                     'Scared to pass flu to a vulnerable relative_encoded','If I got the flu I worry that I would miss out on important activities or events_encoded',
                     'With no flu jab I would feel very vulnerable to the flu_encoded','Vaccination forms part of a healthy lifestyle_encoded']
    for q in questions:
        df_sub = df_new[(df_new.Country == c) & (df_new.Adherence_Score < 70)]
        avg = df_sub[q].mean()
        std_ = df_sub[q].std()
        best_country_theme.loc[rows,'Country'] = c
        if (avg+(std_) > 9.5):
            best_country_theme.loc[rows,q] = 9
        else:
            best_country_theme.loc[rows,q] = avg+(std_)
    rows = rows+1


# In[ ]:

def adherence_calc_2themes_country(country, theme1, theme2):
    df_c = df_new[df_new.Country == country]
    best_cluster_theme_new = best_country_theme[best_country_theme.Country == country]
    df_c_2 = df_c.copy()
    df_c_3 = df_c.copy()
    cluster = '65+'
    
    y1 = list(pam_mapping.loc[pam_mapping['Theme'] == theme1,country])
    y2 = list(pam_mapping.loc[pam_mapping['Theme'] == theme2,country])
    
    
    value_list1 = [0]*len(y1)
    value_list2 = [0]*len(y2)
    
    for i in range(len(y1)):
        value_list1[i] = float(best_cluster_theme_new.loc[:,y1[i]])
    
    for i in range(len(y2)):
        value_list2[i] = float(best_cluster_theme_new.loc[:,y2[i]])
        
    no_change = 0
    
    for c in range(len(value_list1)):
        if (min(df_c.loc[(df_c['Adherence_Score'] < 70),y1[c]]) < value_list1[c]):
            df_c.loc[(df_c[y1[c]] < value_list1[c]) & (df_c['Adherence_Score'] < 70),y1[c]] = value_list1[c]
            no_change = 1
            
    for c in range(len(value_list2)):
        if (min(df_c.loc[(df_c['Adherence_Score'] < 70),y2[c]]) < value_list2[c]):
            df_c.loc[(df_c[y2[c]] < value_list2[c]) & (df_c['Adherence_Score'] < 70),y2[c]] = value_list2[c]
            no_change = 1
    
    calc_theme_score1 = df_c[y1].mean(axis=1).mean()
    act_theme_score1 = df_c_3[y1].mean(axis=1).mean()
    
    calc_theme_score2 = df_c[y2].mean(axis=1).mean()
    act_theme_score2 = df_c_3[y2].mean(axis=1).mean()
    
    
    dummy_data = pd.get_dummies(df_c)

    columns_req = list(df_dummied.columns)

    for i in columns_req:
        if i not in list(dummy_data.columns):
            dummy_data[i] = 0


    exec('df_country = dummy_data[dummy_data.Country_{} == 1]'.format(country))
    df_country2 = df_country[df_country.Adherence_Score < 70]
    data_x_c = df_country2.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country2[['Adherence_Score']]

    exec('y_pred = pd.DataFrame(clf_{}.predict(data_x_c))'.format(country))
    diff_cluster_sum = float(sum(df.loc[(df['Age bucket'] != '65+') & (df['Country'] == country), 'Adherence_Score']))
    diff_cluster_rows = float(len(df.loc[(df['Age bucket'] != '65+') & (df['Country'] == country), 'Adherence_Score']))
    cluster_sum = float(sum(y_pred[0]))+float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score']))
    change_rows = float(len(y_pred[0]))
    non_change_rows = len(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])
    cluster_rows = float(len(df_country.loc[:,'Adherence_Score']))
    total_rows = float(len(df.loc[df['Country'] == country,'Adherence_Score']))
    mean_adh = float(((diff_cluster_sum + cluster_sum)/total_rows))
    clust_adherence = (float(sum(y_pred[0])) + float(sum(df_country.loc[df_country['Adherence_Score'] >= 80,'Adherence_Score'])))/float(len(df_country.loc[:,'Adherence_Score']))
    act_clust_adherence = float(df_c[['Adherence_Score']].mean())
    con_adherence = mean_adh
    act_con_adherence = float(df.loc[df['Country'] == country,'Adherence_Score'].mean())
    #print('{}'.format(country))
    #print('{}'.format(cluster))
    #print('{}'.format(theme))
    #print('{0:0.2f}%'.format(mean_adh))
    return country,cluster,theme1,theme2,calc_theme_score1,act_theme_score1,calc_theme_score2,act_theme_score2,mean_adh,clust_adherence,act_con_adherence,act_clust_adherence
    #,total_rows,diff_cluster_rows,change_rows,non_change_rows


# In[ ]:

themes = ['External Barriers','Doctor relation','Cues to Action','Knowledge on flu','Self efficacy','Vulnerability','Vaccination attitude']
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

