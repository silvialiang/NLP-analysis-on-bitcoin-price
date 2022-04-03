#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 22:06:47 2022
deal with the bitcoin price data and conduct the text analysis
@author: silvia Shuxi Liang u3035878157
"""
import pandas as pd
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
#%%
#a function to extract all full texts from the nested data structure
def dp(string):
    soup = BeautifulSoup(string,'html.parser')
    string = soup.get_text()
    return string
#%%
#a function to get the polarity scores from texts
def sense(string):
    sid = SentimentIntensityAnalyzer()
    return(sid.polarity_scores(string))
#%%
#conduct sentimental analysis for all texts and calculate the daily sentimental scores
filepath = '/Users/silvia/Desktop/NLP/'
full_text = pd.read_csv(filepath + 'full_data_text.csv')
neg = []
pos = []
neu = []
compound = []
for i in tqdm(full_text['full_text']):
    neg.append(sense(str(dp(i)))['neg'])
    neu.append(sense(str(dp(i)))['neu'])
    pos.append(sense(str(dp(i)))['pos'])
    compound.append(sense(str(dp(i)))['compound'])
sense_text = pd.DataFrame({'time':full_text['time'],'neg':neg,'neu':neu,'pos':pos,'compound':compound})
sense_text = sense_text.groupby(['time']).sum()
#%%
#load the btc price data and generate the price and volume momentum for 1 day and 3 days respectively
btc = pd.read_csv(filepath + 'BTC-USD.csv')
#btc momentum = CP-CP(n days ago)
#price_mom_3
pri_mom_3_lead = []
pri_mom_1_lead = []
pri_mom_3_lag = []
pri_mom_1_lag = []
for i in tqdm(range(6,btc.shape[0]-3)):
    pri_mom_3_lead.append(btc['Close'][i+3] - btc['Close'][i])
    pri_mom_3_lag.append(btc['Close'][i-1] - btc['Close'][i-4])
    pri_mom_1_lead.append(btc['Close'][i+1] - btc['Close'][i])
    pri_mom_1_lag.append(btc['Close'][i-1] - btc['Close'][i-2])
btc = btc[6:btc.shape[0]-3]
btc['pri_mom_3_lead'] = pri_mom_3_lead
btc['pri_mom_1_lead'] = pri_mom_1_lead
btc['pri_mom_3_lag'] = pri_mom_3_lag
btc['pri_mom_1_lag'] = pri_mom_1_lag
#select the btc price information based on whether there is a daily sentimental score for the day
index = []
for i in btc['Date']:
    if i in sense_text.index.tolist():
        index.append(True)
    else:
        index.append(False)
btc = btc.loc[index]
#%%
#use machine learning methods to train a model and show how efficient the model is
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
X = sense_text.values
y_3_lead = btc['pri_mom_3_lead'].values
y_3_lag = btc['pri_mom_3_lag'].values
y_1_lead = btc['pri_mom_1_lead'].values
y_1_lag = btc['pri_mom_1_lag'].values
y_close = btc['Close'].values
y_high = btc['High'].values
X_std = preprocessing.scale(X)
y_3_lead_std = preprocessing.scale(y_3_lead)
y_1_lead_std = preprocessing.scale(y_1_lead)
y_1_lag_std = preprocessing.scale(y_1_lag)
y_3_lag_std = preprocessing.scale(y_3_lag)
train_x_1_lead,test_x_1_lead,train_y_1_lead,test_y_1_lead = train_test_split(X_std, y_1_lead_std,test_size = 0.5)
train_x_1_lag,test_x_1_lag,train_y_1_lag,test_y_1_lag = train_test_split(X_std, y_1_lag_std,test_size = 0.5)
train_x_3_lead,test_x_3_lead,train_y_3_lead,test_y_3_lead = train_test_split(X_std, y_3_lead_std,test_size = 0.5)
train_x_3_lag,test_x_3_lag,train_y_3_lag,test_y_3_lag = train_test_split(X_std, y_3_lag_std,test_size = 0.5)
train_x,test_x,train_y,test_y = train_test_split(X_std, y_close,test_size = 0.5)
train_x_h,test_x_h,train_y_h,test_y_h = train_test_split(X_std, y_high,test_size = 0.5)

lasso_1_lead = Lasso(max_iter=10000)
lasso_3_lead = Lasso(max_iter=10000)
lasso_3_lag = Lasso(max_iter=10000)
lasso_1_lag = Lasso(max_iter=10000)
lasso_close = Lasso(max_iter=10000)
lasso_high = Lasso(max_iter=10000)
lasso_1_lead.fit(train_x_1_lead,train_y_1_lead)
lasso_3_lead.fit(train_x_3_lead,train_y_3_lead)
lasso_1_lag.fit(train_x_1_lag,train_y_1_lag)
lasso_3_lag.fit(train_x_3_lag,train_y_3_lag)
lasso_close.fit(train_x,train_y)
lasso_high.fit(train_x_h,train_y_h)
print('lasso_close:',lasso.score(test_x,test_y))
print('lasso_high:',lasso.score(test_x_h,test_y_h))
print('1_lead: ' , lasso_1_lead.score(test_x_1_lead, test_y_1_lead))
print('1_lag: ' , lasso_1_lag.score(test_x_1_lag, test_y_1_lag))
print('3_lead: ' , lasso_3_lead.score(test_x_3_lead, test_y_3_lead))
print('3_lag: ' , lasso_3_lag.score(test_x_3_lag, test_y_3_lag))