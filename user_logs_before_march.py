# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 03:34:24 2017

@author: TomoPC
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:54:11 2017

@author: TomoPC
"""


import numpy as np 
import pandas as pd

import sys
import gc; gc.enable()
import collections

import pandas as pd
import numpy as np
from sklearn import *
from datetime import datetime as dt


#This one is to calculate the 30 day 

userlog_features = ['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']
def make_userlog_features():
   print('loading...')
   infos = {}
#   cutoff1=dt.strptime("20170201", "%Y%m%d")
   cutoff2=dt.strptime("20170301", "%Y%m%d")
   with open('./Data/user_logs.csv') as fd:
       count = 0
       fd.readline()
       for line in fd:
           #print(line)
           pos = line.find(',')
           msid = line[:pos]
           #_, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs = [int(float(value)) for value in line[pos + 1:-1].split(',')]
           splits = line[pos + 1:-1].split(',')
           info = [int(value) for value in splits[:-1]]
           info.append(int(float(splits[-1])))
           #if len(info) != 8:
           #    print('not expect line: %s'%line[:-1])
           #    continue
           
           date=(dt.strptime(str(info[0]), "%Y%m%d"))
           
           if (date <=cutoff2):
#               print(date)
               if msid not in infos:
                   info[0] = 1
                   infos[msid] = info
               else:
                   infos[msid][0] += 1
                   for index in range(1, 8):
                       infos[msid][index] += info[index]
           count += 1
           if count % 100000 == 0:
               print('processed: %d'%count)
#           if count > 100:
#               break
   print('done: %d'%count)

   df_userlog = pd.DataFrame()
   df_userlog['msno'] = infos.keys()
   df_userlog['date_count'] = [infos[key][0] for key in infos.keys()]
   for index, feature in enumerate(userlog_features[1:]):
       if feature == 'total_secs':
           df_userlog[feature] = [infos[key][index+1]/3600 for key in infos.keys()]
       else:
           df_userlog[feature] = [infos[key][index+1] for key in infos.keys()]

   return df_userlog

def make_userlog_features2():
   print('loading...')
   infos = {}
   cutoff1=dt.strptime("20170201", "%Y%m%d")
   cutoff2=dt.strptime("20170301", "%Y%m%d")
   with open('./Data/user_logs_v2.csv') as fd:
       count = 0
       fd.readline()
       for line in fd:
           pos = line.find(',')
           msid = line[:pos]
           #_, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs = [int(float(value)) for value in line[pos + 1:-1].split(',')]
           splits = line[pos + 1:-1].split(',')
           info = [int(value) for value in splits[:-1]]
           info.append(int(float(splits[-1])))
           #if len(info) != 8:
           #    print('not expect line: %s'%line[:-1])
           
           date=(dt.strptime(str(info[0]), "%Y%m%d"))
           #    continue
           if (date <= cutoff2 ):
#               print(date)
               if msid not in infos:
                   info[0] = 1
                   infos[msid] = info
               else:
                   infos[msid][0] += 1
                   for index in range(1, 8):
                       infos[msid][index] += info[index]
           count += 1
           if count % 100000 == 0:
               print('processed: %d'%count)
           #if count > 10000000:
#               break
   print('done: %d'%count)

   df_userlog = pd.DataFrame()
   df_userlog['msno'] = infos.keys()
   df_userlog['date_count'] = [infos[key][0] for key in infos.keys()]
   for index, feature in enumerate(userlog_features[1:]):
       if feature == 'total_secs':
           df_userlog[feature] = [infos[key][index+1]/3600 for key in infos.keys()]
       else:
           df_userlog[feature] = [infos[key][index+1] for key in infos.keys()]

   return df_userlog



user_logs = make_userlog_features()
user_logs2 = make_userlog_features2()

user_logs.to_csv('user_logs_before_march.csv')

user_logs2.to_csv('user_logs2_before_march.csv')