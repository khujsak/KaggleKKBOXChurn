# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:29:09 2017

@author: TomoPC
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

train=pd.read_csv('test_with_members.csv')
train=train.drop(train.columns[0], axis=1)
train_orig=pd.read_csv('test_orig_with_members.csv')
train_orig=train_orig.drop(train_orig.columns[0], axis=1)

date1=dt.date(2017, 1, 1)
date2=dt.date(2017, 3, 1)

path=os.getcwd()

transactions=pd.read_csv(path+'\\Data\\transactions.csv')

transactionsv2=pd.read_csv(path+'\\Data\\transactions_v2.csv')

transactions=pd.concat([transactions, transactionsv2])

del transactionsv2

transactions['payment_method_id'] = transactions['payment_method_id'].astype('int8')
transactions['payment_plan_days'] = transactions['payment_plan_days'].astype('int16')

transactions['plan_list_price'] = transactions['plan_list_price'].astype('int16')
transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype('int16')

transactions['is_auto_renew'] = transactions['is_auto_renew'].astype('int8') # chainging the type to boolean
transactions['is_cancel'] = transactions['is_cancel'].astype('int8')#changing the type to boolean

transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'].astype(str))
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'].astype(str))


transactions=transactions[(transactions['membership_expire_date'] < date2)] 
transactions=transactions[(transactions['membership_expire_date'] >= date1)] 
transactions=transactions[(transactions['membership_expire_date'] > transactions['transaction_date'])]

transactions['membership_expire_date'] = transactions['membership_expire_date'].apply(dt.datetime.toordinal)
transactions['transaction_date'] = transactions['transaction_date'].apply(dt.datetime.toordinal)


#Get Average Time between transactions and Number of Transactions



transac_sort=pd.concat([transactions['msno'], transactions['transaction_date']], axis=1, keys=['msno', 'transaction_date'])

transac_sort=transac_sort.sort_values(['transaction_date'],ascending=True)


results=transac_sort.groupby('msno')['transaction_date'].agg(['max','min', 'count']).reset_index()
results['days_btw_transacs']=((results['max']-results['min'])/results['count']).astype(int)

results=results.rename(columns={'count': 'transaction_count'})
results=results.rename(columns={'min': 'registration_init_time_v2'})

train = pd.merge(left = train,right = results[['msno', 'days_btw_transacs', 'transaction_count', 'registration_init_time_v2']],how = 'left',on=['msno'])

train_orig = pd.merge(left = train_orig,right = results[['msno', 'days_btw_transacs', 'transaction_count', 'registration_init_time_v2']],how = 'left',on=['msno'])



#test = pd.merge(left = test,right = results[['msno', 'days_btw_transacs', 'transaction_count', 'registration_init_time_v2']],how = 'left',on=['msno'])

del transac_sort, results

#Get percent of transactions that were auto_renewed

count_auto_renew=transactions.groupby('msno')['is_auto_renew'].sum().reset_index()

count_auto_renew=count_auto_renew.rename(columns={'is_auto_renew': 'percent_auto_renew'})


train = pd.merge(left = train,right = count_auto_renew,how = 'left',on=['msno'])

train_orig = pd.merge(left = train_orig,right = count_auto_renew,how = 'left',on=['msno'])
#test = pd.merge(left = test,right = count_auto_renew,how = 'left',on=['msno'])

train['percent_auto_renew']=train['percent_auto_renew'].divide(train['transaction_count'])

train_orig['percent_auto_renew']=train_orig['percent_auto_renew'].divide(train_orig['transaction_count'])
#test['percent_auto_renew']=test['percent_auto_renew'].divide(test['transaction_count'])


del count_auto_renew


#Get Last Expiration date



results=transactions.groupby('msno')['membership_expire_date'].agg(['max']).reset_index()


results=results.rename(columns={'max': 'expiration_date'})


train = pd.merge(left = train,right = results, how = 'left',on=['msno'])

train_orig = pd.merge(left = train_orig,right = results, how = 'left',on=['msno'])
#test = pd.merge(left = test,right = results, how = 'left',on=['msno'])

del results

#Get Correct Registration Init Time






#Cumulative difference between list and paid
#Normalized Mean Paid
#Normalized Variance of Actual Paid
#Aggregate grouped by list and paid

transactions['total_membership_time']=transactions['membership_expire_date']-transactions['transaction_date']

transactions['amount_paid_per_day']=(transactions['actual_amount_paid'])/(transactions['payment_plan_days']+1)
transactions['list_price_per_day']=(transactions['plan_list_price'])/(transactions['payment_plan_days']+1)



paidperday=transactions.groupby('msno')['amount_paid_per_day'].agg(['max', 'min', 'mean', 'var'])
listperday=transactions.groupby('msno')['list_price_per_day'].agg(['max', 'min', 'mean'])
totaltime=transactions.groupby('msno')['total_membership_time'].agg(['sum'])

results=totaltime
results=results.drop(['sum'], axis=1)

results['mean_diff_pricevspaid']=(listperday['mean']-paidperday['mean'])


results['mean_amount_paid']=(paidperday['mean'])

#results['var_amount_paid']=(paidperday['var'])

results['cumulative_price_change']=paidperday['max']-paidperday['min']

results['total_paid_membership_time']=totaltime['sum']

results=results.reset_index()



train = pd.merge(left = train,right = results,how = 'left',on=['msno'])

train_orig = pd.merge(left = train_orig,right = results,how = 'left',on=['msno'])

#test = pd.merge(left = test,right = results,how = 'left',on=['msno'])

del results, paidperday, listperday, totaltime


#Get most common payment method



results=transactions.groupby('msno')['payment_method_id'].agg(lambda x: x.value_counts().index[0]).reset_index()

train = pd.merge(left = train,right = results,how = 'left',on=['msno'])

train_orig = pd.merge(left = train_orig,right = results,how = 'left',on=['msno'])

#test = pd.merge(left = test,right = results,how = 'left',on=['msno'])


train.to_csv('test_tr_windowed.csv')
train_orig.to_csv('test_orig_tr_windowed.csv')
#test.to_csv('test_tr.csv')


#Time since last price change


#transac_sort=pd.concat([transactions['msno'], transactions['transaction_date'], transactions['list_price_per_day']], axis=1, keys=['msno', 'transaction_date', 'list_price_per_day'])

#transac_sort=transac_sort.sort_values(['transaction_date'],ascending=True)

#results=transac_sort.groupby('msno')['list_price_per_day'].agg(['max','min', 'count']).reset_index()

