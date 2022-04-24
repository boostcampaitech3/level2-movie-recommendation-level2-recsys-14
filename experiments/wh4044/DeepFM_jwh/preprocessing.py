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

# 1. Rating df 생성
base_dir = '/opt/ml/input'
rating_data = base_dir + "/data/train/train_ratings.csv"

raw_rating_df = pd.read_csv(rating_data)

raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)

users = set(raw_rating_df['user'])
items = set(raw_rating_df['item'])
print('user 수 :', len(users))
print('item 수 :', len(items))

#2. Genre df 생성
genre_data = base_dir + "/data/train/genres.tsv"

raw_genre_df = pd.read_csv(genre_data, sep='\t')
raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경

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

# 4. Negative instance 생성
print("Create Nagetive instances")
num_negative = 100
user_group_dfs = list(raw_rating_df.groupby('user')['item']) # 유저별로 별점을 주었던 item 정리

first_row = True
user_neg_dfs = pd.DataFrame()
df_list = []
for u, u_items in tqdm(user_group_dfs):
    u_items = set(u_items)
    # 전체 items 리스트에서 user가 봤던 u_items를 제외한 리스트에서 50개를 랜덤으로 추출
    i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)

    i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
    if first_row == True:
        user_neg_dfs = i_user_neg_df
        first_row = False

    else:
        df_list.append(i_user_neg_df)

user_neg_dfs = pd.concat([user_neg_dfs, *df_list], axis = 0, sort=False)

# 기존의 df와 concat한다.
raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)

# 5. pos + neg 데이터프레임과 genre 데이터프레임 merge
joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner')
joined_rating_df = pd.merge(joined_rating_df, year_df, left_on='item', right_on='item', how='inner')

# 6. user, item을 zero-based index로 mapping => 0부터 각 개수만큼 reindexing
users = list(set(joined_rating_df.loc[:,'user']))
users.sort()
items =  list(set((joined_rating_df.loc[:, 'item'])))
items.sort()
years =  list(set((joined_rating_df.loc[:, 'year'])))
years.sort()
genres =  list(set((joined_rating_df.loc[:, 'genre'])))
genres.sort()

if len(users)-1 != max(users):
    users_dict = {users[i]: i for i in range(len(users))}
    joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
    users = list(set(joined_rating_df.loc[:,'user']))
    
if len(items)-1 != max(items):
    items_dict = {items[i]: i for i in range(len(items))}
    joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
    items =  list(set((joined_rating_df.loc[:, 'item'])))
    
if len(years)-1 != max(years):
    years_dict = {years[i]: i for i in range(len(years))}
    joined_rating_df['year']  = joined_rating_df['year'].map(lambda x : years_dict[x])
    years =  list(set((joined_rating_df.loc[:, 'year'])))

joined_rating_df = joined_rating_df.sort_values(by=['user'])
joined_rating_df.reset_index(drop=True, inplace=True)

data = joined_rating_df
data.to_csv("data.csv", index=False)
print('*****data saved done!*****')