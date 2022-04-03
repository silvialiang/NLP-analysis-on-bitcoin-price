#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:55:50 2022
Get the full text of the news from the url via web scraping and store the full text with other factors in 'full_text_data.csv'
@author: silvia Shuxi Liang u3035878157
"""
from multiprocessing import Pool
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
#%%
def wbscrape(url):
    cookies = '_ga=GA1.2.971513925.1644827667; _vwo_uuid_v2=DAC772C345373E4A02174A2C21AAEEA31|160c5b6e7ed94635ec5d54b4a8f4293f; __cfruid=28bcfbbca999c097567b7477dc32a87f773ac6d7-1645685062; notice_behavior=implied,us; com.silverpop.iMAWebCookie=c1565554-d81e-05c9-0d45-d7d752736a57; _gcl_au=1.1.503287163.1645704100; _gid=GA1.2.310669002.1645704100; _vis_opt_s=1|; _vis_opt_test_cookie=1; _vwo_uuid=D99985D36F515F3F8ECB3986932E1BD72; _vwo_ds=3:a_0,t_0:0$1645704098:46.8602561:::1_0:0; fulltextShowAll=YES; oneSearchTZ=480; EBSESSIONID=936af21e0cf94482bc9c17d08b9b2071; EBUQUSER=936af21e0cf94482bc9c17d08b9b2071; recentInstitutions=14548; AppVersion=r2022.1.0.3411; iv=14331880-9128-4699-8879-ec2d67ca5572; JSESSIONID=FCE0CAC09E48CA69FF3759C7C74E51A9.i-06d23bde53a76e771; authenticatedBy=TOKEN; authThrough=TgqQKQiChzyAIuSNYDDiC+YvutMUjIYXlIljdsfnUOz36BrDw6oe+kykPqnVHff7ABMJlKdgKdGLxnhgcEmZFTrNnYCWmj3qjR4jp6NayOgBPUADzXRf1CQ83POls1Wn; authSub="Fri Feb 25 09:35:09 UTC 2022"; AWSELB=297563AD084DD59DC6CF4FD721088009D00FA6D7445EAB7F063AF050D8E0692D855579BEAACB2BF505E5A7A75733FD2296E5DE3A6E79CFE708506514DB673C939DF50FB6218CE712CF5E25F6E98443292A4389924C; osTimestamp=1645783353.887; availability-zone=us-east-1f'
    headers = {
        'authority':'www.proquest.com',
        'path':'/news/docview/1785702196/fulltext/900906DB42954156PQ/2?accountid=14548',
        'method':'GET',
        'Referer':'https://www.proquest.com/news/docview/1785702196/fulltext/900906DB42954156PQ/2?accountid=10134',
        'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
        'sec-fetch-mode':'navigate',
        'upgrade-insecure-requests':'1',
        'sec-ch-ua':'" Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"',
        'sec-ch-ua-mobile':'?0',
        'sec-fetch-dest':'document',
        'sec-fetch-site':'cross-site',
        'sec-ch-ua-platform':'"macOS"',
        'sec-fetch-user':'?1',
        'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-language':'zh-CN,zh;q=0.9'
        }
    cookies_jar = requests.cookies.RequestsCookieJar()
    for cookie in cookies.split(';'):
        key, value = cookie.split('=', 1)
        cookies_jar.set(key,value)
    try:
        res = requests.get(url, headers = headers, cookies = cookies_jar)
        s = BeautifulSoup(res.text,'lxml')
        k = str(s.find('text').find_all('p'))
    except:
        k = np.nan
    return k
        
#%%
#multiprocessing for web scraping the full text for those url links in the data file
if __name__ == '__main__':
    filepath = '/Users/silvia/Desktop/NLP/'
    data = pd.read_csv(filepath + 'data.csv') #the data.csv is generated by url_selected.py
    full_text = []
    for i in tqdm(range(270)):
        with Pool(8) as p:
            full_text.extend(list(tqdm(p.imap(wbscrape, data['url'][i*100:(i+1)*100].values.tolist()),total = 100)))
            time.sleep(20)
    data = data[:27000] # slice the data file to keep it consistently with the web scraping results
    data.insert(6,"full_text",full_text)
    data = data.loc[:,['title','time','source','url','full_text']]
    data = data.dropna()#drop all the rows where no valid full text data is
    data.to_csv(filepath + 'full_data_text.csv')
    #store the data in 'full_data_text.csv' 
    #it contains all text data we needed later in nested data structure
    