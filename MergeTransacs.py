# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:56:27 2017

@author: TomoPC
"""

transactions=pd.read_csv('./Data/transactions.csv')
transactionsv2=pd.read_csv('./Data/transactions_v2.csv')


frames = [transactions, transactionsv2]
train = pd.concat(frames)

train.to_csv('transactionsmerged.csv')