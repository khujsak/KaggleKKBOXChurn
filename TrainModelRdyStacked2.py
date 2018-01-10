# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:18:51 2017

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
from sklearn.metrics import roc_curve, auc,recall_score,precision_score,log_loss                                                            
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


train=pd.read_csv('train_rdy.csv')
train_orig=pd.read_csv('train_orig_rdy.csv')
train_january=pd.read_csv('train_january_rdy.csv')
train_january['is_churn']=train_january['is_churn'].astype(int)

train_december=pd.read_csv('train_december_rdy.csv')
train_december['is_churn']=train_december['is_churn'].astype(int)

test=pd.read_csv('test_rdy.csv')



frames = [train, train_january, train_december]
train = pd.concat(frames)

del train_january, train_december, train_orig

train=train.drop(train.columns[0], axis=1)
test=test.drop(test.columns[0], axis=1)



train['membership_expire_date']=(pd.to_datetime(train['membership_expire_date'])).map(dt.datetime.toordinal).apply(int)
test['membership_expire_date']=(pd.to_datetime(test['membership_expire_date'])).map(dt.datetime.toordinal).apply(int)



train['transaction_date']=(pd.to_datetime(train['transaction_date'])).map(dt.datetime.toordinal).apply(int)
test['transaction_date']=(pd.to_datetime(test['transaction_date'])).map(dt.datetime.toordinal).apply(int)


#train=train.drop('membership_expire_date', axis=1)
#test=test.drop('membership_expire_date', axis=1)



#train=train.drop('expiration_date', axis=1)
#test=test.drop('expiration_date', axis=1)


#train=train.drop('transaction_date', axis=1)
#test=test.drop('transaction_date', axis=1)

train=train.drop('secs_per_day_lifetime', axis=1)
test=test.drop('secs_per_day_lifetime', axis=1)
train['payment_method_change']=((train.payment_method_id_x != train.payment_method_id_y)).astype(np.int8)
test['payment_method_change']=(test.payment_method_id_x != test.payment_method_id_y).astype(np.int8)
train=train.drop('payment_method_id_x', axis=1)
test=test.drop('payment_method_id_x', axis=1)
train=train.drop('payment_method_id_y', axis=1)
test=test.drop('payment_method_id_y', axis=1)

train['is_taipei']=((train.city==1)).astype(int)

test['is_taipei']=((test.city==1)).astype(int)

train=train.drop('city', axis=1)
test=test.drop('city', axis=1)

train=train.fillna('median')
test=test.fillna('median')
#dfcity=pd.get_dummies(train['city'])
#train=train.drop(['city'], axis=1)
#train=train.join(dfcity)
#
#del dfcity

#dfpaymenty=pd.get_dummies(train['payment_method_id_y'])
#dfpaymenty.columns = [str(col) + '_payment' for col in dfpaymenty.columns]
#train=train.drop(['payment_method_id_y'], axis=1)
#train=train.join(dfpaymenty)
#
#del dfpaymenty


#dfcity=pd.get_dummies(test['city'])
#test=test.drop(['city'], axis=1)
#test=test.join(dfcity)
#
#del dfcity

#dfpaymenty=pd.get_dummies(test['payment_method_id_y'])
#dfpaymenty.columns = [str(col) + '_payment' for col in dfpaymenty.columns]
#test=test.drop(['payment_method_id_y'], axis=1)
#test=test.join(dfpaymenty)
#
#del dfpaymenty
#

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking





cols = [c for c in train.columns if c not in ['is_churn','msno']]

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test



    
    
x_train, labels, x_test = model_1(train[cols], train['is_churn'], test[cols])
    
preds = model_2()


test['is_churn'] = preds.clip(0.+1e-15, 1-1e-15)

test[['msno','is_churn']].to_csv('submission.csv.gz', index=False, compression='gzip')


import lightgbm as lgb


rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
SEED=1

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer








def model_1(train, labels, test):
    train = np.array(train)
    test = np.array(test)
    labels = np.array(labels)

    ntrain = train.shape[0]
    ntest = test.shape[0]

    kf = KFold(n_splits=5,
               shuffle=True, random_state=2017)

    
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }


    xgb_params = {}
    xgb_params['tree_method'] = 'gpu_hist'
    xgb_params['objective'] =  'binary:logistic'
    xgb_params['eval_metric'] =  'logloss'
    xgb_params['learning_rate'] = 0.1
    xgb_params['max_depth'] = 8
    xgb_params['subsample'] = 0.8
    xgb_params['colsample_bytree'] = 0.7
    xgb_params['colsample_bylevel'] = 0.7
    xgb_params['silent'] = 1
    xgb_params['scale_pos_weight'] = 9

    xg = XgbWrapper(seed=2017, params=xgb_params)
    lg = LgbWrapper(seed=2017, params=lgb_params)

    lg_oof_train, lg_oof_test = get_oof(lg, ntrain, ntest, kf, train, labels, test)
    xg_oof_train, xg_oof_test = get_oof(xg, ntrain, ntest, kf, train, labels, test)



    print("XG-CV: {}".format(mean_squared_error(labels, xg_oof_train)))
    print("LG-CV: {}".format(mean_squared_error(labels, lg_oof_train)))

    x_train = np.concatenate((xg_oof_train, lg_oof_train), axis=1)
    x_test = np.concatenate((xg_oof_test, lg_oof_test), axis=1)

    np.save(arr=x_train, file='x_concat_train.npy')
    np.save(arr=x_test, file='x_concat_test.npy')
    np.save(arr=labels, file='y_labels.npy')

    return x_train, labels, x_test








def model_2():
    train = np.load('x_concat_train.npy')
    labels = np.load('y_labels.npy')
    test = np.load('x_concat_test.npy')

    dtrain = xgb.DMatrix(train, label=labels)
    dtest = xgb.DMatrix(test)

    xgb_params = {}
    xgb_params['tree_method'] = 'gpu_hist'
    xgb_params['objective'] =  'binary:logistic'  
    xgb_params["eta"] = 0.1
    xgb_params["subsample"] = 0.9
    xgb_params["silent"] = 1
    xgb_params["max_depth"] = 5
    xgb_params['eval_metric'] =  'logloss'
    xgb_params['min_child_weight'] = 10
    xgb_params['seed'] = 2017
    xgb_params['scale_pos_weight'] = 9

    res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=5, seed=2017, stratified=False,
                 early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]

    print('')
    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
    bst = xgb.train(xgb_params, dtrain, best_nrounds)

    preds = bst.predict(dtest)
    return preds



import pandas as pd
import numpy as np

from sklearn import preprocessing
#from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb


class XgbWrapper(object):
    def __init__(self, seed=2017, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 400)

    def train(self, xtra, ytra, xte, yte):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        dvalid = xgb.DMatrix(xte, label=yte)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
            watchlist, early_stopping_rounds=10)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

class LgbWrapper(object):
    def __init__(self, seed=2017, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 400)

    def train(self, xtra, ytra, xte, yte):
        ytra = ytra.ravel()
        yte = yte.ravel()
        dtrain = lgb.Dataset(xtra, label=ytra)
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(x)


def get_oof(clf, ntrain, ntest, kf, train, labels, test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))
    
    i=0
    for (train_index, test_index) in kf.split(train, labels):
        x_tr = train[train_index]
        y_tr = labels[train_index]
        x_te = train[test_index]
        y_te = labels[test_index]

        clf.train(x_tr, y_tr, x_te, y_te)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(test)
        i+1

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
