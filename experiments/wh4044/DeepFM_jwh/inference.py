import csv
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import *

print("*****Start Inference*****")
# 1. Rating df 생성
base_dir = '/opt/ml/input'
rating_data = base_dir + "/data/train/train_ratings.csv"

raw_rating_df = pd.read_csv(rating_data)

user_uni = sorted(raw_rating_df['user'].unique())
item_uni = sorted(raw_rating_df['item'].unique())

user2idx, idx2user, item2idx, idx2item = dict(), dict(), dict(), dict()

for i in range(len(user_uni)):
    user2idx[user_uni[i]] = i
    idx2user[i] = user_uni[i]

for i in range(len(item_uni)):
    item2idx[item_uni[i]] = i
    idx2item[i] = item_uni[i]

raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)
print('***Step 1 Done***')
# -----------------------------------------------------------------------------------------------

#2. Genre df 생성
genre_data = base_dir + "/data/train/genres.tsv"

raw_genre_df = pd.read_csv(genre_data, sep='\t')
raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

genre_uni = sorted(raw_genre_df['genre'].unique())
genre2idx, idx2genre = dict(), dict()
for i in range(len(genre_uni)):
    genre2idx[genre_uni[i]] = i
    idx2genre[i] = genre_uni[i]

raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre2idx[x]) #genre id로 변경
print('***Step 2 Done***')
# -----------------------------------------------------------------------------------------------

#3. Year df 생성
year_data = pd.read_csv(base_dir + '/data/train/years.tsv', sep='\t')
# title df에서 year 정보 확인하기
title_data = pd.read_csv(base_dir + '/data/train/titles.tsv', sep='\t')

genre_items = set(sorted(raw_genre_df['item'].unique()))
year_items = set(sorted(year_data['item'].unique()))
year_ = sorted(list(genre_items.symmetric_difference(year_items)))

non_year = title_data.loc[title_data['item'].isin(year_)]
non_year['year'] = non_year['title'].apply(lambda x : int(x[-5:-1]))
non_year = non_year.drop(columns=['title'])

year_df = pd.concat([year_data, non_year]).reset_index(drop=True)

year_uni = sorted(year_df['year'].unique())
year2idx = dict()
for i in range(len(year_uni)):
    year2idx[year_uni[i]] = i
# print('year2idx len', len(year2idx))
# time.sleep(10)
# -----------------------------------------------------------------------------------------------
print('***Step 3 Done***')


# Inference 준비---------------------------------------------------------------------------------
df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner')
df = pd.merge(df, year_df, left_on='item', right_on='item', how='inner')

df2 = df[['item','genre', 'year']].drop_duplicates()
df2['item'] = df2['item'].apply(lambda x : item2idx[x])
df2['year'] = df2['year'].apply(lambda x : year2idx[x])
df2 = df2.sort_values("item")
df2 = df2.reset_index(drop=True)

n = len(user_uni) + len(item_uni)
user_col = np.array([0]*len(df2))
item_col = np.array(df2['item'] + len(user_uni))
genre_col = np.array(df2['genre'] + n)
year_col = np.array(df2['year'] + n+18)

temp = zip(user_col, item_col, genre_col, year_col)
temp = list(map(list, temp))
temp = torch.tensor(temp)

df['user'] = df['user'].apply(lambda x : user2idx[x])
df['item'] = df['item'].apply(lambda x : item2idx[x])
df['year'] = df['year'].apply(lambda x : year2idx[x])

user_group = list(df.groupby('user')['item'])

user_pos = dict()
for i, j in tqdm(user_group):
    j = list(j.values)
    user_pos[i] = j

model = torch.load('best_model.pt')

user_df = pd.DataFrame(columns=["item","rating"])
user_df['item'] = item_uni
# -----------------------------------------------------------------------------------------------
print('***Step 4 Done***')

# Inference--------------------------------------------------------------------------------------
device = torch.device('cuda')
output_li = []
for user in tqdm(user_uni):
    user_id = user2idx[user]
    temp[:,0] = user_id
    temp = temp.to(device)
    
    with torch.no_grad():
        output = model(temp)
        user_df['rating'] = output.cpu().numpy()
        pred = user_df.sort_values('rating', ascending=False)[:3000]        
        neg = []
        idx = 0
        while len(neg) < 10:
            item = pred[idx:idx+1]['item'].values[0]
            if item2idx[item] in user_pos[user_id]:
                idx += 1
                continue
            else:
                neg.append(item)
                idx += 1
        output_li.append(pd.DataFrame({'user':[user]*10, 'item':neg}))      

output = pd.concat(output_li, axis = 0, sort=False)
output.to_csv('output.csv', index=False)

print('Inference Done!! output saved************')
# -----------------------------------------------------------------------------------------------
