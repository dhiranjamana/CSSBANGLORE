from google.cloud import bigquery
import pandas as pd

client = bigquery.Client.from_service_account_json('...\\Google Project-080164a110bb.json')


query_2009 ="""SELECT a.stn, b.country , b.state , b.name , b.lat , b.lon , a.year , a.mo , avg(a.temp) as mon_avg_temp 
FROM [bigquery-public-data:noaa_gsod.gsod2009] as a  
join [bigquery-public-data:noaa_gsod.stations] as b  
on a.stn=b.usaf  where country in ('CH','FR','MX','UK','US') 
group by a.stn, b.country , b.state , b.name , b.lat , b.lon , a.year , a.mo"""

weather_data_2009 = pd.read_gbq( query_2009, project_id='vast-incline-169817')

weather_data_2009.to_gbq("ANTHEM.Weather_Test", project_id='vast-incline-169817', chunksize=10000, verbose=True, reauth=False, if_exists='fail', private_key='...\\Google Project-4fe99986324f.json')
