
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


df = pd.read_csv("C:\Users\css113429\Desktop\Google Sanofi\simulation_new_theme_5.csv")

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




#exec('{}_coefs = pd.DataFrame()'.format(i))
#exec('{}_coefs = imp'.format(i))
#exec('{}_coefs.to_csv("C:\Users\css113429\Desktop\Google Sanofi\{}.csv")'.format(i,i))


#y_pred = clf.predict(X_test)

#sim_data = pd.DataFrame()

#print("Overall Accuracy = %s" %metrics.accuracy_score(y_pred,y_test))
#print("Classification Matrix")
#print(metrics.classification_report(y_pred,y_test))


#Overall Model
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



themes = ['Barriers','Cues to action','Doctor relationship','Information on Flu','Self efficacy','Threat perception','Vaccination attitude']

        


rating_answer = {
        0: {'label': '0%'},
        1: {'label': '10%'},
        2: {'label': '20%'},
        3: {'label': '30%'},
        4: {'label': '40%'},
        5: {'label': '50%'},
        6: {'label': '60%'},
        7: {'label': '70%'},
        8: {'label': '80%'},
        9: {'label': '90%'},
        10: {'label': '100%'},
        11: {'label': 'No change'}
    }

percent_answer = {
        0: {'label': '0%'},
        1: {'label': '10%'},
        2: {'label': '20%'},
        3: {'label': '30%'},
        4: {'label': '40%'},
        5: {'label': '50%'},
        6: {'label': '60%'},
        7: {'label': '70%'},
        8: {'label': '80%'},
        9: {'label': '90%'},
        10: {'label': '100%'}
    }


paths = [{'label': 'Knowledge', 'value':0},
         {'label': 'Self efficacy', 'value':1},
         {'label': 'Response efficacy', 'value':2},
         {'label': 'Cues to Action', 'value':3},
         {'label': 'Perceived Severity', 'value':4},
         {'label': 'Perceived Vulnerability', 'value':5},
         {'label': 'Barriers', 'value':6}]
#percent_answer = ["No change",0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#rating_answer = ["No change",0,1,2,3,4,5,6,7,8,9,10]


app = dash.Dash()
#auth = dash_auth.BasicAuth(
#    app,
#    VALID_USERNAME_PASSWORD_PAIRS
#)
####Background colours
colors = {
    'background': '#F0F3F4',
    'text': '#111111'
}


##App front end
app.layout = html.Div(style={'backgroundColor': colors['background'], 'zoom' : '66.5%'}, children=[
    html.H1(
        children='Business Impact Calculator',
        style={
            'textAlign': 'center',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'color': '#21618C',
            'margin-right':'100px',
            'font-size':'60px',
            'heigth':'100px'
        }
    ),
    
    html.Button('Reset' ,id='button', n_clicks=0, style={'float':'right','margin-top':'50px'}),
    
    html.Div(id='country-adherence-name',
             children="Actual adherence score for the country: ",
             style={
            'textAlign': 'right',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'25px',
            'margin-left':'5px',
            'color': colors['text'],
            'width':'620px','height':'100px', 'display': 'inline-block'
            
        }),
    
    html.Div(id='country-adherence',
             children="XX%",
             style={
            'textAlign': 'left',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'50px',
            'margin-left':'50px','margin-right':'40px','width':'100px','height':'100px',
            'color': '#2196F3','display': 'inline-block'
        }),
    
    html.Div(id='calc-adherence-name',
             children="Calculated adherence score for the country: ",
             style={
            'textAlign': 'right',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'25px',
            'margin-left':'200px','width':'700px','height':'100px',
            'color': colors['text'], 'display': 'inline-block'
        }),
    
    html.Div(id='calc-adherence',
             children="XX%",
             style={
            'textAlign': 'left',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'50px',
            'margin-left':'50px','width':'100px','height':'100px',
            'margin-right':'240px',
            'color':'#2196F3' ,'display': 'inline-block'
        }),
    
    
    html.Div(id='cluster-adherence-name',
             children="Actual adherence score for the cluster: ",
             style={
            'textAlign': 'right',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'25px',
            'margin-left':'5px',
            'color': colors['text'],
            'width':'620px','height':'100px', 'display': 'inline-block'
            
        }),
    
    html.Div(id='cluster-adherence',
             children="XX%",
             style={
            'textAlign': 'left',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'50px',
            'margin-left':'50px','margin-right':'40px','width':'100px','height':'100px',
            'color': '#2196F3','display': 'inline-block'
        }),
    
    html.Div(id='calc-adherence-name-cluster',
             children="Calculated adherence score for the cluster: ",
             style={
            'textAlign': 'right',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'25px',
            'margin-left':'200px','width':'700px','height':'100px',
            'color': colors['text'], 'display': 'inline-block'
        }),
    
    html.Div(id='calc-adherence-cluster',
             children="XX%",
             style={
            'textAlign': 'left',
            'font-family':'Helvetica',
            'font-weight':'bold',
            'font-size':'50px',
            'margin-left':'50px','width':'100px','height':'100px',
            'margin-right':'240px',
            'color':'#2196F3' ,'display': 'inline-block'
        }),
    
    
    html.Div([html.H3(
        children='Country',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'center',
            'color': '#FDFEFE',
            'border-radius': '15px 15px 15px 15px',
            'backgroundColor': '#21618C',
            'box-shadow': '0px 8px 16px 0px rgba(0,0,0,0.2)'
        }
    ), 
        dcc.Dropdown(
                id='countries-dropdown',
                options=[{'label': k, 'value': k} for k in countries],
                value='US'
            )],style={'width': '200px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-left':'100px', 'margin-right':'20px'} ),
    
    
    html.Div([html.H3(
        children='Cluster',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'center',
            'color': '#FDFEFE',
            'border-radius': '15px 15px 15px 15px',
            'backgroundColor': '#21618C',
            'box-shadow': '0px 8px 16px 0px rgba(0,0,0,0.2)'
        }
    ), 
        dcc.Dropdown(
                id='cluster-dropdown',
                options=[{'label': k, 'value': k} for k in clusters],
                value='US1'
            )],style={'width': '600px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-left':'20px', 'margin-right':'50px'} ),
                
    html.Div([
        dcc.Graph(id='intent-graph')],
                style={'width': '50px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                       'margin-left':'10px','margin-right':'300px', 'margin-top' : '20px'}),

    html.Div([
        dcc.Graph(id='afford-graph')],
                style={'width': '50px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                       'margin-left':'10px','margin-right':'200px', 'margin-top' : '20px'}),

    html.Div([
        dcc.Graph(id='access-graph')],
                style={'width': '50px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                       'margin-left':'10px','margin-right':'200px', 'margin-top' : '20px'}),
    
    html.Div([
        dcc.Graph(id='time-graph')],
                style={'width': '50px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                       'margin-left':'10px','margin-right':'200px', 'margin-top' : '20px'}),

#    html.Div([html.H6(
#        id='diabetic_percent',
#        children='Diabetics - ',
#        style={
#            'font-family' : 'Helvetica',
#            'textAlign': 'left',
#            'color': colors['text'],
#            'font-size':'20px',
#            'font-weight':'bold',
#            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
#                'margin-left':'20px', 'margin-top':'80px', 'margin-right':'0px'
#        }
#    )]),
#
#    
#    html.Div([html.H6(
#        id='avg_age',
#        children='Average age - ',
#        style={
#            'font-family' : 'Helvetica',
#            'textAlign': 'left',
#            'color': colors['text'],
#            'font-size':'20px',
#            'font-weight':'bold',
#            'width': '300px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
#                'margin-left':'20px', 'margin-top':'80px', 'margin-right':'20px'
#        }
#    )]),
    
    
    
    html.Div([html.H6(
        id='theme1-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'180px', 'margin-right':'1300px'
        }
    )]),
    


    html.Div([html.H6(
        id='theme1-name',
        children='Information on Flu',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'180px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='info-knowledge-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='info-social-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'50px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='info-ha-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'50px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='info-tv-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'50px','margin-top':'15px'}),
    
    
    html.Div([dcc.RangeSlider(
            id='knowledge_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='social_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'150px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='ha_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'160px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='tv_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'160px'} ),
    
    
    html.Div([html.H6(
        id='theme2-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'1300px'
        }
    )]),
    


    html.Div([html.H6(
        id='theme2-name',
        children='Vaccination attitude',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='vac-favour-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='vac-scared-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='vac-open-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px', 'margin-right':'100px'}),
    
    
    
    html.Div([dcc.RangeSlider(
            id='favour_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='scared_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'250px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='open_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'100px','margin-top':'200px','margin-left':'260px'} ),
    
    
    html.Div([html.H6(
        id='theme3-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'1300px'
        }
    )]),
    
    
    html.Div([html.H6(
        id='theme3-name',
        children='Lack of barriers',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='bar-afford-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='bar-access-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='bar-time-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px', 'margin-right':'100px'}),
    
    
    
    html.Div([dcc.RangeSlider(
            id='afford_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='access_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'250px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='time_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'100px','margin-top':'200px','margin-left':'260px'} ),
    
    
    html.Div([html.H6(
        id='theme4-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'1300px'
        }
    )]),
    
    
    html.Div([html.H6(
        id='theme4-name',
        children='Doctor relationship',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='doc-reco-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='doc-inv-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='doc-info-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px', 'margin-right':'100px'}),
    
    
    
    html.Div([dcc.RangeSlider(
            id='reco_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='inv_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'250px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='info_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'100px','margin-top':'200px','margin-left':'260px'} ),
    
    
    html.Div([html.H6(
        id='theme5-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'1300px'
        }
    )]),
    
    
    html.Div([html.H6(
        id='theme5-name',
        children='Cues to action',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='cue-informed-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='cue-relatives-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='cue-dec-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px', 'margin-right':'100px'}),
    
    
    
    html.Div([dcc.RangeSlider(
            id='informed_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='relatives_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'250px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='dec_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'100px','margin-top':'200px','margin-left':'260px'} ),
    
    
    html.Div([html.H6(
        id='theme6-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'1300px'
        }
    )]),
    
    
    html.Div([html.H6(
        id='theme6-name',
        children='Threat perception',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='threat-severe-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='threat-sick-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='threat-scared-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px', 'margin-right':'100px'}),
    
    
    
    html.Div([dcc.RangeSlider(
            id='severe_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='sick_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'250px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='fluscared_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'100px','margin-top':'200px','margin-left':'260px'} ),
    
    html.Div([html.H6(
        id='theme7-score',
        children='Theme Score:',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '200px','height': '20px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'1300px'
        }
    )]),
    
    
    html.Div([html.H6(
        id='theme7-name',
        children='Self efficacy',
        style={
            'font-family' : 'Helvetica',
            'textAlign': 'left',
            'color': colors['text'],
            'font-size':'20px',
            'font-weight':'bold',
            'width': '500px','height': '5px', 'display': 'inline-block', 'padding': '0 20', 'float':'right',
                'margin-left':'0px', 'margin-top':'40px', 'margin-right':'130px'
        }
    )]),


    html.Div([
        dcc.Graph(id='self-know-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'100px','margin-top':'30px'}),
    
    
    html.Div([
        dcc.Graph(id='self-health-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px'}),
    
    html.Div([
        dcc.Graph(id='self-action-graph')
    ], style={'width': '500px','height': '10px', 'display': 'inline-block',
                        'padding': '0 20', 'float':'left','margin-left':'150px','margin-top':'15px', 'margin-right':'100px'}),
    
    
    
    html.Div([dcc.RangeSlider(
            id='know_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'90px'} ),
    
    
    html.Div([dcc.RangeSlider(
            id='health_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'15px','margin-top':'200px','margin-left':'250px'} ),
    
    
    
    html.Div([dcc.RangeSlider(
            id='action_slider',
            min=0,
            step=0.5,
            max=10,
            value=[0,10],
            marks = {0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                        5: {'label': '5'},
                        6: {'label': '6'},
                        7: {'label': '7'},
                        8: {'label': '8'},
                        9: {'label': '9'},
                        10: {'label': '10'}
                        }
        )],style={'width': '370px','height': '50px', 'display': 'inline-block', 'padding': '0 20', 'float':'left',
                'margin-right':'100px','margin-top':'200px','margin-left':'260px'} ),
    
    
    
])


@app.callback(
    dash.dependencies.Output('cluster-dropdown', 'options'),
    [dash.dependencies.Input('countries-dropdown', 'value')])
def update_adherence(country):
    if (country == "Overall"):
        clusters_new = list(df['Cluster'].unique())
    else:
        df_c = df[df.Country == country]
        clusters_new = list(df_c['Cluster'].unique())

    return [{'label': k, 'value': k} for k in clusters_new]


@app.callback(
    dash.dependencies.Output('cluster-dropdown', 'value'),
    [dash.dependencies.Input('countries-dropdown', 'value')])
def update_adherence(country):
    if (country == "Overall"):
        clusters_new = list(df['Cluster'].unique())
    else:
        df_c = df[df.Country == country]
        clusters_new = list(df_c['Cluster'].unique())

    return clusters_new[0]


@app.callback(
    dash.dependencies.Output('country-adherence', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value')])
def update_adherence(country):
    if (country == "Overall"):
        df_c = df_new.copy()
    else:
        df_c = df_new[df_new.Country == country]
        
    y_test = df_c[['Adherence_Score']]
    
    return '{0:0.2f}%'.format(round(y_test.mean(),2))


@app.callback(
    dash.dependencies.Output('cluster-adherence', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    if (country == "Overall"):
        df_c = df_new[(df['Cluster'] == cluster)]
    else:
        df_c = df_new[df_new.Country == country]
        df_c = df_c[(df_c['Cluster'] == cluster)]

    y_test = df_c[['Adherence_Score']]
    
    return '{0:0.2f}%'.format(round(y_test.mean(),2))

@app.callback(
        dash.dependencies.Output('intent-graph','figure'),
        [dash.dependencies.Input('countries-dropdown', 'value'),
         dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    if (country == "Overall"):
        df_c = df[(df['Cluster'] == cluster)]
    else:
        df_c = df[df.Country == country]
        df_c = df_c[(df_c['Cluster'] == cluster)]
    
    cont_table = df_c.groupby(by = ['Intent to vaccinate'])['Intent to vaccinate'].count()
    categ = list(cont_table.index)
    valu = list(cont_table)
    
    colors = ["#32CD32","#FFA500","#ff0000"]
    for i in range(3):
        if (categ[i] == 'High'):
            colors[i] = "#32CD32"
        elif (categ[i] == 'Medium'):
            colors[i] = "#FFA500"
        else:
            colors[i] = "#ff0000"
    
    return {
        'data':[
            dict(
                labels=categ,
                values = valu,
                hoverinfo='label+percent',
                marker={'colors' : colors},
                hole = 0.4,
                type = "pie"
            )
        ],
        'layout':go.Layout(
            width=250,
            height=150,
            showlegend=False,
            title = 'Intent to Vaccinate',
            margin=go.Margin(
                l=0,
                r=0,
                b=0,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
        dash.dependencies.Output('afford-graph','figure'),
        [dash.dependencies.Input('countries-dropdown', 'value'),
         dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    if (country == "Overall"):
        df_c = df[(df['Cluster'] == cluster)]
    else:
        df_c = df[df.Country == country]
        df_c = df_c[(df_c['Cluster'] == cluster)]
    
    cont_table = df_c.groupby(by = ['Affordability'])['Affordability'].count()
    categ = list(cont_table.index)
    valu = list(cont_table)
    colors = [""]*len(categ)
    for i in range(len(categ)):
        if (categ[i] == 'Yes'):
            colors[i] = "#32CD32"
        elif (categ[i] == 'Dont Know'):
            colors[i] = "#FFA500"
        else:
            colors[i] = "#ff0000"
    
    return {
        'data':[
            dict(
                labels=categ,
                values = valu,
                hoverinfo='label+percent',
                marker={'colors' : colors},
                hole = 0.4,
                type = "pie"
            )
        ],
        'layout':go.Layout(
            width=250,
            height=150,
            showlegend=False,
            title = 'Affordability',
            margin=go.Margin(
                l=0,
                r=0,
                b=0,
                t=30,
                pad=4
                )
        )
    }

@app.callback(
        dash.dependencies.Output('access-graph','figure'),
        [dash.dependencies.Input('countries-dropdown', 'value'),
         dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    if (country == "Overall"):
        df_c = df[(df['Cluster'] == cluster)]
    else:
        df_c = df[df.Country == country]
        df_c = df_c[(df_c['Cluster'] == cluster)]
    
    cont_table = df_c.groupby(by = ['Accessibility'])['Accessibility'].count()
    categ = list(cont_table.index)
    valu = list(cont_table)
    
    colors = ["#32CD32","#FFA500","#ff0000"]
    for i in range(3):
        if (categ[i] == 'High'):
            colors[i] = "#32CD32"
        elif (categ[i] == 'Medium'):
            colors[i] = "#FFA500"
        else:
            colors[i] = "#ff0000"
    
    return {
        'data':[
            dict(
                labels=categ,
                values = valu,
                hoverinfo='label+percent',
                marker={'colors' : colors},
                hole = 0.4,
                type = "pie"
            )
        ],
        'layout':go.Layout(
            width=250,
            height=150,
            showlegend=False,
            title = 'Accessibility',
            margin=go.Margin(
                l=0,
                r=0,
                b=0,
                t=30,
                pad=4
                )
        )
    }

@app.callback(
        dash.dependencies.Output('time-graph','figure'),
        [dash.dependencies.Input('countries-dropdown', 'value'),
         dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    if (country == "Overall"):
        df_c = df[(df['Cluster'] == cluster)]
    else:
        df_c = df[df.Country == country]
        df_c = df_c[(df_c['Cluster'] == cluster)]
    
    cont_table = df_c.groupby(by = ['Time availability'])['Time availability'].count()
    categ = list(cont_table.index)
    valu = list(cont_table)
    colors = [""]*len(categ)
    for i in range(len(categ)):
        if (categ[i] == 'Yes'):
            colors[i] = "#32CD32"
        elif (categ[i] == 'Dont Know'):
            colors[i] = "#FFA500"
        else:
            colors[i] = "#ff0000"
    
    return {
        'data':[
            dict(
                labels=categ,
                values = valu,
                hoverinfo='label+percent',
                marker={'colors' : colors},
                hole = 0.4,
                type = "pie"
            )
        ],
        'layout':go.Layout(
            width=250,
            height=150,
            showlegend=False,
            title = 'Time availability',
            margin=go.Margin(
                l=0,
                r=0,
                b=0,
                t=30,
                pad=4
                )
        )
    }



#theme1 score
@app.callback(
    dash.dependencies.Output('theme1-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Information on Flu_score'].mean(),1))


@app.callback(
    dash.dependencies.Output('info-knowledge-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Knowledge of flu'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            xaxis=dict(
                range=[0, 10]
            ),
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('info-social-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Trust info provided by social media'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('info-ha-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Trust the info provided by Health Authorities'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }

@app.callback(
    dash.dependencies.Output('info-tv-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Trust the info provided by TV/radio/newspapers'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=400,
            height=200,
            title=y,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('theme2-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Vaccination attitude_score'].mean(),1))






@app.callback(
    dash.dependencies.Output('vac-favour-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Favouring vaccination'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('vac-scared-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Not scared of getting a flu jab'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('vac-open-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Open to flu jabs'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }
        
        
@app.callback(
    dash.dependencies.Output('theme3-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Barriers_score'].mean(),1))






@app.callback(
    dash.dependencies.Output('bar-afford-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Flu jab affordability'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('bar-access-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Flu jab accessibility'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('bar-time-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Time availability for flu jab'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('theme4-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Doctor relationship_score'].mean(),1))






@app.callback(
    dash.dependencies.Output('doc-reco-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Doctor recommends flu jab'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('doc-inv-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Doctor involvement in health decisions'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('doc-info-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Trust in doctor about flu jab info'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }
        

@app.callback(
    dash.dependencies.Output('theme5-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Cues to action_score'].mean(),1))






@app.callback(
    dash.dependencies.Output('cue-informed-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Informed to get the flu jab'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('cue-relatives-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Relatives recommends flu jab'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('cue-dec-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Relatives help in health decisions'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('theme6-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Threat perception_score'].mean(),1))






@app.callback(
    dash.dependencies.Output('threat-severe-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Flu could make me severely ill'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('threat-sick-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Perception of sick time from flu'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('threat-scared-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Scared of getting the flu'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('theme7-score', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_adherence(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    return 'Theme Score: {0:0.1f}'.format(round(df_c['Self efficacy_score'].mean(),1))






@app.callback(
    dash.dependencies.Output('self-know-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Know enough about the flu jab'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('self-health-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Self health rating'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('self-action-graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value')])
def update_graph(country, cluster):
    df_c = df_new[df_new.Country == country]
    df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = 'Taking action to protect against flu'
    
    df_c[[y]] = df_c[[y]].round(2).astype(int)    
    
    
    df2 = df_c.groupby([y, "Adherence_Score_2"])["Adherence_Score_2"].count().reset_index(name="count")
    df3  = df2.pivot_table('count',y, "Adherence_Score_2")
    new= list(df3.sum(axis=1))
    df3 = df3.div(new,axis=0)    
    return {
        'data':[
            go.Bar(
                x=list(df3.index),
                y=list(df3[0]),
                name="Adherence - No",
                marker=go.Marker(
                    color='rgb(75, 196, 213)'
                )
            ),
            go.Bar(
                x=list(df3.index),
                y=list(df3[1]),
                name="Adherence - Yes",
                marker=go.Marker(
                    color='rgb(1, 128, 181)'
                )
            )
        ],
        'layout':go.Layout(
            barmode = "stack",
            width=300,
            height=200,
            title=y,
            showlegend=False,
            margin=go.Margin(
                l=20,
                r=10,
                b=20,
                t=30,
                pad=4
                )
        )
    }


@app.callback(
    dash.dependencies.Output('calc-adherence-cluster', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value'),
     dash.dependencies.Input('knowledge_slider','value'),
     dash.dependencies.Input('social_slider','value'),
     dash.dependencies.Input('ha_slider','value'),
     dash.dependencies.Input('tv_slider','value'),
     dash.dependencies.Input('favour_slider','value'),
     dash.dependencies.Input('scared_slider','value'),
     dash.dependencies.Input('open_slider','value'),
     dash.dependencies.Input('afford_slider','value'),
     dash.dependencies.Input('access_slider','value'),
     dash.dependencies.Input('time_slider','value'),
     dash.dependencies.Input('reco_slider','value'),
     dash.dependencies.Input('inv_slider','value'),
     dash.dependencies.Input('info_slider','value'),
     dash.dependencies.Input('informed_slider','value'),
     dash.dependencies.Input('relatives_slider','value'),
     dash.dependencies.Input('dec_slider','value'),
     dash.dependencies.Input('severe_slider','value'),
     dash.dependencies.Input('sick_slider','value'),
     dash.dependencies.Input('fluscared_slider','value'),
     dash.dependencies.Input('know_slider','value'),
     dash.dependencies.Input('health_slider','value'),
     dash.dependencies.Input('action_slider','value')])
def update_graph(country, cluster,knowledge_slider, social_slider, ha_slider, tv_slider, favour_slider, scared_slider, open_slider, afford_slider, access_slider, time_slider, reco_slider, inv_slider, info_slider, informed_slider, relatives_slider, dec_slider, severe_slider, sick_slider, fluscared_slider, know_slider, health_slider, action_slider):
    if (country == "Overall"):
        df_c = df_new[(df_new['Cluster'] == cluster)]
    else:
        df_c = df_new[df_new.Country == country]
        df_c = df_c[(df_c['Cluster'] == cluster)]
    
    y = ['Knowledge of flu', 'Trust info provided by social media', 'Trust the info provided by Health Authorities', 'Trust the info provided by TV/radio/newspapers', 'Favouring vaccination', 'Not scared of getting a flu jab', 'Open to flu jabs', 'Flu jab affordability', 'Flu jab accessibility', 'Time availability for flu jab', 'Doctor recommends flu jab', 'Doctor involvement in health decisions', 'Trust in doctor about flu jab info', 'Informed to get the flu jab', 'Relatives recommends flu jab', 'Relatives help in health decisions', 'Flu could make me severely ill', 'Perception of sick time from flu', 'Scared of getting the flu', 'Know enough about the flu jab', 'Self health rating', 'Taking action to protect against flu']
    
    value_list = [knowledge_slider[0], social_slider[0], ha_slider[0], tv_slider[0], favour_slider[0], scared_slider[0], open_slider[0], afford_slider[0], access_slider[0], time_slider[0], reco_slider[0], inv_slider[0], info_slider[0], informed_slider[0], relatives_slider[0], dec_slider[0], severe_slider[0], sick_slider[0], fluscared_slider[0], know_slider[0], health_slider[0], action_slider[0]]
    
    no_change = 0
    
    for c in range(len(value_list)):
        if (min(df_c[y[c]]) < value_list[c]):
            df_c.loc[df_c[y[c]] < value_list[c],y[c]] = value_list[c]
            no_change = 1
    
    df_c = df_c.drop(['Cluster','Adherence_Score_2','Barriers_score','Cues to action_score','Doctor relationship_score',
                      'Information on Flu_score','Self efficacy_score','Threat perception_score','Vaccination attitude_score'],axis=1)
    
    dummy_data = pd.get_dummies(df_c)
    
    columns_req = list(df_dummied.columns)
    
    for i in columns_req:
        if i not in list(dummy_data.columns):
            dummy_data[i] = 0

    
    exec('df_country = dummy_data[dummy_data.Country_{} == 1]'.format(country))
    data_x_c = df_country.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country[['Adherence_Score']]
    
    if (no_change==0):
        return '{0:0.2f}%'.format(round(df_c[['Adherence_Score']].mean(),2))
    else:
        exec('y_pred = pd.DataFrame(clf_{}.predict(data_x_c))'.format(country))
        return '{0:0.2f}%'.format(float(y_pred.mean()))



@app.callback(
    dash.dependencies.Output('calc-adherence', 'children'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cluster-dropdown', 'value'),
     dash.dependencies.Input('knowledge_slider','value'),
     dash.dependencies.Input('social_slider','value'),
     dash.dependencies.Input('ha_slider','value'),
     dash.dependencies.Input('tv_slider','value'),
     dash.dependencies.Input('favour_slider','value'),
     dash.dependencies.Input('scared_slider','value'),
     dash.dependencies.Input('open_slider','value'),
     dash.dependencies.Input('afford_slider','value'),
     dash.dependencies.Input('access_slider','value'),
     dash.dependencies.Input('time_slider','value'),
     dash.dependencies.Input('reco_slider','value'),
     dash.dependencies.Input('inv_slider','value'),
     dash.dependencies.Input('info_slider','value'),
     dash.dependencies.Input('informed_slider','value'),
     dash.dependencies.Input('relatives_slider','value'),
     dash.dependencies.Input('dec_slider','value'),
     dash.dependencies.Input('severe_slider','value'),
     dash.dependencies.Input('sick_slider','value'),
     dash.dependencies.Input('fluscared_slider','value'),
     dash.dependencies.Input('know_slider','value'),
     dash.dependencies.Input('health_slider','value'),
     dash.dependencies.Input('action_slider','value')])
def update_graph(country, cluster,knowledge_slider, social_slider, ha_slider, tv_slider, favour_slider, scared_slider, open_slider, afford_slider, access_slider, time_slider, reco_slider, inv_slider, info_slider, informed_slider, relatives_slider, dec_slider, severe_slider, sick_slider, fluscared_slider, know_slider, health_slider, action_slider):
    if (country == "Overall"):
        df_c = df_new[(df_new['Cluster'] == cluster)]
        df_c_2 = df_new.copy()
        
    else:
        df_c = df_new[df_new.Country == country]
        df_c_2 = df_c.copy()
        df_c = df_c[(df_c['Cluster'] == cluster)]
        
    
    y = ['Knowledge of flu', 'Trust info provided by social media', 'Trust the info provided by Health Authorities', 'Trust the info provided by TV/radio/newspapers', 'Favouring vaccination', 'Not scared of getting a flu jab', 'Open to flu jabs', 'Flu jab affordability', 'Flu jab accessibility', 'Time availability for flu jab', 'Doctor recommends flu jab', 'Doctor involvement in health decisions', 'Trust in doctor about flu jab info', 'Informed to get the flu jab', 'Relatives recommends flu jab', 'Relatives help in health decisions', 'Flu could make me severely ill', 'Perception of sick time from flu', 'Scared of getting the flu', 'Know enough about the flu jab', 'Self health rating', 'Taking action to protect against flu']
    
    value_list = [knowledge_slider[0], social_slider[0], ha_slider[0], tv_slider[0], favour_slider[0], scared_slider[0], open_slider[0], afford_slider[0], access_slider[0], time_slider[0], reco_slider[0], inv_slider[0], info_slider[0], informed_slider[0], relatives_slider[0], dec_slider[0], severe_slider[0], sick_slider[0], fluscared_slider[0], know_slider[0], health_slider[0], action_slider[0]]
    
    no_change = 0
    
    for c in range(len(value_list)):
        if (min(df_c[y[c]]) < value_list[c]):
            df_c.loc[df_c[y[c]] < value_list[c],y[c]] = value_list[c]
            no_change = 1
    
    df_c = df_c.drop(['Cluster','Adherence_Score_2','Barriers_score','Cues to action_score','Doctor relationship_score',
                       'Information on Flu_score','Self efficacy_score','Threat perception_score','Vaccination attitude_score'],axis=1)
    
    dummy_data = pd.get_dummies(df_c)
    
    columns_req = list(df_dummied.columns)
    
    for i in columns_req:
        if i not in list(dummy_data.columns):
            dummy_data[i] = 0

    
    exec('df_country = dummy_data[dummy_data.Country_{} == 1]'.format(country))
    data_x_c = df_country.drop(['Adherence_Score'],axis=1)
    data_y_c = df_country[['Adherence_Score']]
    
    if (no_change==0):
        return '{0:0.2f}%'.format(round(df_c_2[['Adherence_Score']].mean(),2))
    else:
        exec('y_pred = pd.DataFrame(clf_{}.predict(data_x_c))'.format(country))
        diff_cluster_sum = float(sum(df_c_2.loc[df_c_2['Cluster'] != cluster,'Adherence_Score']))
        cluster_sum = float(sum(y_pred[0]))
        total_rows = float(len(df_c_2.loc[:,'Adherence_Score']))
        mean_adh = float(((diff_cluster_sum + cluster_sum)/total_rows))
        return '{0:0.2f}%'.format(mean_adh)
      
    

    
app.scripts.config.serve_locally = True
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

if __name__ == '__main__':
    app.run_server(debug=True)

