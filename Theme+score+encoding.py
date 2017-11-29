
# coding: utf-8

# In[57]:

import pandas as pd
import numpy as np
from scipy import stats


# In[83]:

df = pd.read_csv("C:\Users\css113429\Desktop\Google Sanofi\\Combined_data_cleaned_v4_drilldown.csv")
variable_encoding = pd.read_csv("C:\Users\css113429\Desktop\Google Sanofi\\variable_encoding.csv")


# In[84]:

for i in list(variable_encoding["Column"].unique()):
    data_to_merge = variable_encoding.loc[variable_encoding["Column"]==i,["Column_Value","Encoding"]]
    if (df[i].dtype != np.object):
        data_to_merge["Column_Value"] = data_to_merge["Column_Value"].astype(int)
    exec('data_to_merge.columns = ["{}","{}_encoded"]'.format(i,i))
    df = pd.merge(df,data_to_merge, on=[i], how="left")
    exec('df.drop(["{}"], axis=1)'.format(i))


# In[86]:

df.to_csv("C:\Users\css113429\Desktop\Google Sanofi\\encoded_data_vaxitrend.csv")
df

