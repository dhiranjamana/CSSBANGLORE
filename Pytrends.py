
# coding: utf-8

# In[176]:

from pytrends.request import TrendReq
from datetime import datetime
import time
import pandas as pd
# # Login to Google. Only need to run this once, the rest of requests will use the same session.
# pytrend = TrendReq()

# # Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
# pytrend.build_payload(kw_list=['pizza', 'bagel'])

# # Interest Over Time
# interest_over_time_df = pytrend.interest_over_time()
# print(interest_over_time_df.head())

# # Interest by Region
# interest_by_region_df = pytrend.interest_by_region()
# print(interest_by_region_df.head())

# # Related Queries, returns a dictionary of dataframes
# related_queries_dict = pytrend.related_queries()
# print(related_queries_dict)

# # Get Google Hot Trends data
# trending_searches_df = pytrend.trending_searches()
# print(trending_searches_df.head())

# # Get Google Top Charts
# top_charts_df = pytrend.top_charts(cid='actors', date=201611)
# print(top_charts_df.head())

# # Get Google Keyword Suggestions
# suggestions_dict = pytrend.suggestions(keyword='pizza')
# print(suggestions_dict)


# In[113]:

pytrend = TrendReq()
pytrend.build_payload(kw_list=['flu', 'flu shot'],timeframe='2013-01-01 2015-12-31', geo='GB')
# pytrends = build_payload(kw_list, cat=0, timeframe='2016-12-14 2017-01-25', geo='US-NY', gprop='')
pytrend.interest_over_time()


# In[114]:

dates = pytrend.interest_over_time()


# In[19]:

dates.index[0].strftime("%Y-%m-%d %H:%M:%S").split()[0]


# In[115]:

date_strs = [i.strftime("%Y-%m-%d %H:%M:%S").split()[0] for i in dates.index]


# In[177]:

date_strs


# In[59]:

pytrend = TrendReq()
pytrend.build_payload(kw_list=['flu', 'flu shot','cdc flu shot','cold flu','cold symptoms'],timeframe='2013-01-06 2015-01-13', geo='EU')
frame = pytrend.interest_by_region()
frame


# In[26]:

frame["Date-Range"] = '2013-01-06 2015-01-13'


# In[254]:

data_lists = []
for i,elem in enumerate(date_strs):
    if i != 155:
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=['diabetes', 'type 2 diabetes','symptoms diabetes','type 1 diabetes','what is diabetes'],timeframe=date_strs[i]+" "+date_strs[i+1], geo='GB')
        frame = pytrend.interest_by_region()
        frame["Date-Range"] = date_strs[i]+" "+date_strs[i+1]
        data_lists.append(frame)
        


# In[255]:

df = pd.concat(data_lists,axis=0)


# In[256]:

df.to_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBFirst_5.csv",encoding = 'utf-8')


# In[257]:

data_lists = []
for i,elem in enumerate(date_strs):
    if i != 155:
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=['sugar diabetes', 'symptoms of diabetes','diabetes test','diabetes causes','diabetes treatment'],timeframe=date_strs[i]+" "+date_strs[i+1], geo='GB')
        frame = pytrend.interest_by_region()
        frame["Date-Range"] = date_strs[i]+" "+date_strs[i+1]
        data_lists.append(frame)
        


# In[258]:

df = pd.concat(data_lists,axis=0)


# In[259]:

df.to_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBSecond_5.csv",encoding = 'utf-8')


# In[260]:

data_lists = []
for i,elem in enumerate(date_strs):
    if i != 155:
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=['ketoacidosis', 'diabetic ketoacidosis','ketoacidosis symptoms','ketoacidosis causes','glycemia'],timeframe=date_strs[i]+" "+date_strs[i+1], geo='GB')
        frame = pytrend.interest_by_region()
        frame["Date-Range"] = date_strs[i]+" "+date_strs[i+1]
        data_lists.append(frame)


# In[261]:

df = pd.concat(data_lists,axis=0)


# In[262]:

df.to_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBThird_5.csv",encoding = 'utf-8')


# In[126]:

data_lists = []
for i,elem in enumerate(date_strs):
    if i != 155:
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=['flu virus', 'how long does it take to recover from flu','how long flu last','nhs flu','reaction to flu shot'],timeframe=date_strs[i]+" "+date_strs[i+1], geo='GB')
        frame = pytrend.interest_by_region()
        frame["Date-Range"] = date_strs[i]+" "+date_strs[i+1]
        data_lists.append(frame)


# In[127]:

df = pd.concat(data_lists,axis=0)


# In[128]:

df.to_csv("C:\Users\css112831\Desktop\Google Trends\UK_Fourth_5.csv")


# In[129]:

data_lists = []
for i,elem in enumerate(date_strs):
    if i != 155:
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=['side effects from flu jab', 'side effects of flu jab','side effects of flu shot','what is flu','what is the flu shot'],timeframe=date_strs[i]+" "+date_strs[i+1], geo='GB')
        frame = pytrend.interest_by_region()
        frame["Date-Range"] = date_strs[i]+" "+date_strs[i+1]
        data_lists.append(frame)


# In[130]:

df = pd.concat(data_lists,axis=0)


# In[131]:

df.to_csv("C:\Users\css112831\Desktop\Google Trends\UK_Fifth_5.csv")


# In[263]:

US_states = pd.read_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBFirst_5.csv")


# 

# In[264]:

States = list(set(US_states["geoName"]))


# In[265]:

States


# In[266]:

len(States)


# In[267]:

test_frame = pd.DataFrame({"States":States})


# In[268]:

State_lists = []
for i,elem in enumerate(date_strs):
    if i != 155:
        test_frame = pd.DataFrame({"geoName":States})
        test_frame["Date-Range"] = date_strs[i]+" "+date_strs[i+1]
        State_lists.append(test_frame)


# In[269]:

State_lists[2]


# In[270]:

df_unique = pd.concat(State_lists,axis=0)


# In[271]:

df_unique = df_unique.reset_index()


# In[272]:

df_unique


# In[273]:

df_first = pd.read_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBFirst_5.csv")
df_second = pd.read_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBSecond_5.csv")
df_third = pd.read_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBThird_5.csv")
# df_fourth = pd.read_csv("C:\Users\css112831\Desktop\Google Trends\UK_Fourth_5.csv")
# df_fifth = pd.read_csv("C:\Users\css112831\Desktop\Google Trends\UK_Fifth_5.csv")


# In[274]:

first_join = pd.merge(left = df_unique ,right =df_first,how= 'left',on = ["geoName","Date-Range"])


# In[275]:

Second_join = pd.merge(left = first_join ,right =df_second,how= 'left',on = ["geoName","Date-Range"])


# In[276]:

Third_join = pd.merge(left = Second_join ,right =df_third,how= 'left',on = ["geoName","Date-Range"])


# In[143]:

Fourth_join = pd.merge(left = Third_join ,right =df_fourth,how= 'left',on = ["geoName","Date-Range"])


# In[144]:

Fifth_join = pd.merge(left = Fourth_join ,right =df_fifth,how= 'left',on = ["geoName","Date-Range"])


# In[277]:

Third_join = Third_join.fillna(0)


# In[278]:

Third_join.to_csv("C:\Users\css112831\Desktop\Google Trends\UK_DBMaster.csv",encoding = 'utf-8')

