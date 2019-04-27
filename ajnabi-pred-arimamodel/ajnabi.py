# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:16:27 2019

@author: hp
"""



import mysql.connector as sql
import pandas as pd
import numpy as np

#creating connection
db_connection = sql.connect(host='localhost', database='jtsboard', user='root', password='')
db_cursor = db_connection.cursor()

#fetching sinle "customer_histories table from mysql database
db_cursor.execute('SELECT id, user_id, customer_id, service_price, date FROM customer_histories')
sql_data1 = db_cursor.fetchall()

#creating dataframe for "customer_histories"
df1=pd.DataFrame(sql_data1, columns=["id", "user_id","customer_id","service_price", "date"])
#print(df1.head())

#converting "custmer histories" dataframe to CSV
df1.to_csv('custhist.csv', index=False)

data=pd.read_csv("custhist.csv")

data.shape
user_102=data[data['user_id']==102.0]

user_102=user_102.reset_index()
del user_102['index']
user_102.head()


grouped=user_102.groupby(user_102['customer_id'])

# user_102.groupby(user_102['customer_id']).groups

user_102.shape

user_102.dtypes

a=user_102['date'][0]

user_102['date']=pd.to_datetime(user_102['date'])


user_102['Month']=user_102['date'].apply(lambda x:x.month)
user_102['Year']=user_102['date'].apply(lambda x:x.year)
user_102['Day']=user_102['date'].apply(lambda x:x.day)

user_102.isnull().sum()
# Here, there are 6 rows which have missing values,so I am dropping it because without date there is no significance of it
user_102=user_102.dropna(subset=['date'])
user_102['date'].isnull().sum()
user_102.isnull().sum()



user_102.Month=user_102.Month.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))


g = user_102[['customer_id','Month','Year']].groupby(['Month','Year']).agg('count')

g=pd.DataFrame(g)
g.columns=['Total_visits']

g=g.reset_index()
g=g.sort_values('Year')

import matplotlib.pyplot as plt
import datetime

g['Month']=g['Month'].apply(lambda x : datetime.date(1900,x, 1).strftime('%B'))

f, ax = plt.subplots(figsize=(18,5))
barlist=plt.bar(g['Month'],g['Total_visits'])


customer_wise_data=user_102.groupby(['Month','Year']).agg('count')['date']

customer_wise_data=pd.DataFrame(customer_wise_data)


train=customer_wise_data[:7]
test=customer_wise_data[7:]

from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(train,order=(0,0,1))
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train)+1)
print(predictions)

model=ARIMA(customer_wise_data,order=(0,0,1))

model=ARIMA(customer_wise_data,order=(1,0,0))

model_fit = model.fit()


predictions = model_fit.predict(start=len(customer_wise_data), end=len(customer_wise_data)+2)
print("Prediction for three Months------------\n",round(predictions))