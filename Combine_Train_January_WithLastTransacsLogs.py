# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:12:40 2017

@author: TomoPC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:48:41 2017

@author: TomoPC
"""

import pandas as pd
import numpy as np
import datetime as dt

#Train Original

#Use transactions_last_february.csv
#Use user_logs_february_to_march.csv
#Use user_logs2_february_to_march.csv

#Use user_logs_before_february.csv



#Process and Load user logs

user_logs = pd.read_csv('./Logs_Processed/user_logs_december_to_january.csv')
user_logs2=pd.read_csv('./Logs_Processed/user_logs2_january_to_february.csv')


frames = [user_logs, user_logs2]
result = pd.concat(frames)

result=result.drop(result.columns[0], axis=1)
user_logs=user_logs.drop(user_logs.columns[0], axis=1)

result=result.groupby(['msno']).sum()
result=result.reset_index()


user_logs_last=result



user_logs_last.columns = [str(col) + '_last_entry' for col in user_logs_last.columns]


user_logs_last = user_logs_last.rename(columns={'msno_last_entry': 'msno'})

del result


user_logs = pd.read_csv('./Logs_Processed/user_logs_before_january.csv')
#user_logs2=pd.read_csv('./Logs_Processed/user_logs2_before_january.csv')

#frames = [user_logs, user_logs2]
#result = pd.concat(frames)

result=user_logs

result=result.drop(result.columns[0], axis=1)
user_logs=user_logs.drop(user_logs.columns[0], axis=1)

result=result.groupby(['msno']).sum()
result=result.reset_index()

user_logs_history=result


del result


#Now load Transactions



transactions_last=pd.read_csv('./Trans_Processed/transactions_last_january.csv')
transactions_last=transactions_last.drop(transactions_last.columns[0], axis=1)



#Now load train file

train_w=pd.read_csv('train_january_tr_windowed.csv')
train_w=train_w.drop(train_w.columns[0], axis=1)
train_w=train_w.drop(train_w.columns[1:9], axis=1)
train_w=train_w.drop(train_w.columns[3], axis=1)
train_w=train_w.drop(train_w.columns[4], axis=1)


train_w.columns = [str(col) + '_window' for col in train_w.columns]


train_w = train_w.rename(columns={'msno_window': 'msno'})



train=pd.read_csv('train_january_tr.csv')



train=train.drop(train.columns[0], axis=1)
#train_orig=train_orig.drop(['city'], axis=1)
#train_orig=train_orig.drop(['registered_via'], axis=1)
#train_orig=train_orig.drop(['registration_init_time'], axis=1)
#train_orig=train_orig.drop(['bd'], axis=1)
#train=train.drop(['expiration_date_x'], axis=1)


#train['expiration_date']=train['expiration_date_y']
#train=train.drop(['expiration_date_y'], axis=1)

train.male = train.male.fillna(value=0)
train.female = train.female.fillna(value=0)
train.unknown_gender = train.unknown_gender.fillna(value=1)


train= pd.merge(left = train, right = user_logs_history,how = 'left',on=['msno'])
train=pd.merge(left=train, right=user_logs_last, how='left', on=['msno'])
train=pd.merge(left=train, right=transactions_last, how='left', on=['msno'])
train=pd.merge(left=train, right=train_w, how='left', on=['msno'])




# Now Process



train['secs_per_day_lifetime']=train['total_secs']/train['total_paid_membership_time']

train=train.drop(['total_secs'], axis=1)


train['secs_per_day_last']=train['total_secs_last_entry']/train['payment_plan_days']

train=train.drop(['total_secs_last_entry'], axis=1)



#Num Unique per day


train['unique_per_day']=train['num_unq']/train['total_paid_membership_time']


train['date_count_last_entry']=train['date_count_last_entry'].fillna(0)



train['num_25_last_entry']=(train['num_25_last_entry']/train['num_25'])*(train['date_count']/train['date_count_last_entry'])
train['num_100_last_entry']=(train['num_100_last_entry']/train['num_100'])*(train['date_count']/train['date_count_last_entry'])
train['num_985_last_entry']=(train['num_985_last_entry']/train['num_985'])*(train['date_count']/train['date_count_last_entry'])
train['num_50_last_entry']=(train['num_50_last_entry']/train['num_50'])*(train['date_count']/train['date_count_last_entry'])
train['num_75_last_entry']=(train['num_75_last_entry']/train['num_75'])*(train['date_count']/train['date_count_last_entry'])
train['num_unq_last_entry']=(train['num_unq_last_entry']/train['num_unq'])*(train['date_count']/train['date_count_last_entry'])
     
    
     

#Ratio of all songs to unique songs
#

train['unique_over_all']=train['num_unq']/(train['num_25']+train['num_50']+train['num_75']+train['num_985']+train['num_100'])


#Normalizing the nums

train_tot=(train['num_50']+train['num_75']+train['num_985']+train['num_100']+train['num_25'])

train['num_25']=train['num_25']/train_tot

train['num_50']=train['num_50']/train_tot

train['num_75']=train['num_75']/train_tot

train['num_985']=train['num_985']/train_tot

train['num_100']=train['num_100']/train_tot

     
train=train.drop(['num_unq'], axis=1)

train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)


train.to_csv('train_january_rdy.csv')