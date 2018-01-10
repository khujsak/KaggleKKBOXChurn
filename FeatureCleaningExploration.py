# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:28:32 2017

@author: TomoPC
"""

import os
from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
import datetime as dt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc,recall_score,precision_score                                                             
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

train=pd.read_csv('train_rdy.csv')
train_orig=pd.read_csv('train_orig_rdy.csv')
train_january=pd.read_csv('train_january_rdy.csv')
train_january['is_churn']=train_january['is_churn'].astype(int)

train_december=pd.read_csv('train_december_rdy.csv')
train_december['is_churn']=train_december['is_churn'].astype(int)

test=pd.read_csv('test_rdy.csv')



frames = [train, train_january, train_december]
train = pd.concat(frames)

#del train_january, train_december, train_orig

train=train.drop(train.columns[0], axis=1)
test=test.drop(test.columns[0], axis=1)



train['membership_expire_date']=(pd.to_datetime(train['membership_expire_date'])).map(dt.datetime.toordinal).apply(int)
test['membership_expire_date']=(pd.to_datetime(test['membership_expire_date'])).map(dt.datetime.toordinal).apply(int)



train['transaction_date']=(pd.to_datetime(train['transaction_date'])).map(dt.datetime.toordinal).apply(int)
test['transaction_date']=(pd.to_datetime(test['transaction_date'])).map(dt.datetime.toordinal).apply(int)

