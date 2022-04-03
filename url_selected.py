#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 03:17:52 2022
@author: silvia Shuxi Liang u3035878157
to select the urls from 2015 to 2020 and their other factors('title','time','snippet','source')
"""
import pandas as pd
from langdetect import detect
from tqdm import tqdm
from dateutil.parser import parse
import numpy as np
#%%
#to store new
truncatedResultsTitle = []
titleAuthorETC = []
snippet = []
addFlashPageParameterformat_fulltext = []
source = []
full_text = []
index = []
#%%
#data processing
#load all files to process
filepath = '/Users/silvia/Desktop/NLP/'
df_2010_2015 = pd.read_csv(filepath + 'bitcoin_2010_2015_proquest.csv')
df_2016 = pd.read_csv(filepath + 'bitcoin_2016_proquest.csv') 
df_2017_1 = pd.read_csv(filepath + 'bitcoin_2017_01_11_proquest.csv')
df_2017_2 = pd.read_csv(filepath + 'bitcoin_2017_11_12_proquest.csv')
df_2018_1 = pd.read_csv(filepath + 'bitcoin_2018_01_04_proquest.csv')
df_2018_2 = pd.read_csv(filepath + 'bitcoin_2018_04_12_proquest.csv')
df_2019 = pd.read_csv(filepath + 'bitcoin_2019_proquest.csv')
df_2020 = pd.read_csv(filepath + 'bitcoin_2020_proquest.csv')
df_2021_1 = pd.read_csv(filepath + 'bitcoin_2021_01_03_proquest.csv')
df_2021_2 = pd.read_csv(filepath + 'bitcoin_2021_04_proquest.csv')

df = [df_2010_2015,df_2016,df_2017_1,df_2017_2,df_2018_1,df_2018_2,df_2019,df_2020,df_2021_1,df_2021_2]
#%%
#drop all rows which is not english and store the factors needed in the lists
for file in df:
    for row in tqdm(range(file.shape[0])):
        try:
            if detect(file.iloc[row]['snippet']) == 'en':
                addFlashPageParameterformat_fulltext.append(file.iloc[row]['addFlashPageParameterformat_fulltext href'])
                truncatedResultsTitle.append(file.iloc[row]['truncatedResultsTitle'])
                titleAuthorETC.append(str(file.iloc[row]['titleAuthorETC']) + str(file.iloc[row]['titleAuthorETC 2']))
                snippet.append(file.iloc[row]['snippet'])
                source.append(file.iloc[row]['source-type-label'])   
        except:
            continue
#%%
#time_p() function is to find all the date information and convert them in the format of '%Y-%m-%d'
#preprocess the data before 2015 to np.nan to delete them conveniently later
year = ['2015','2016','2017','2018','2019','2020','2021']
def time_p(a, i=0):
    try:
        a = parse(a[a.find(year[i])-9:a.find(year[i])+4]).strftime('%Y-%m-%d')
        return a
    except:
        if i != 7:
            return time_p(a, i+1)
        else:
            return np.nan
#%%
data = {'title' : truncatedResultsTitle,
        'time' : titleAuthorETC,
        'snippet' : snippet,
        'source' : source,
        'url' : addFlashPageParameterformat_fulltext}
data = pd.DataFrame.from_dict(data)
data = data[~data['time'].str.contains('Vol', regex=False)]
data['time'] = data['time'].apply(time_p)
data = data.dropna()
data = data.drop_duplicates(subset=['snippet'])
data.to_csv(filepath + 'data.csv')
#store the data in the NLP file in 'data.csv'