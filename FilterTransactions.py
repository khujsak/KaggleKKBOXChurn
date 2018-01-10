# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:05:39 2017

@author: TomoPC
"""

from datetime import date
import datetime

date=datetime.date(2017, 1, 1)


transactions = pd.read_csv('./Data/transactions.csv')
transactions = pd.concat((transactions, pd.read_csv('./Data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

transactions['membership_expire_date']=pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d')

transactions['transaction_date']=pd.to_datetime(transactions['transaction_date'], format='%Y%m%d')



test=transactions['transaction_date']>transactions['membership_expire_date']


transactions=transactions[(transactions['membership_expire_date'] < date)] 

transactions=transactions[(transactions['membership_expire_date'] > transactions['transaction_date'])]


counts = pd.DataFrame(transactions['msno'].value_counts().reset_index())
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')

transactions['discount'] = transactions['plan_list_price'] - transactions['actual_amount_paid']
transactions['is_discount'] = transactions.discount.apply(lambda x: 1 if x > 0 else 0)


transactions['membership_days'] = transactions['membership_expire_date'].subtract(transactions['transaction_date']).dt.days.astype(int)

transactions.to_csv('transactions_last_december.csv')