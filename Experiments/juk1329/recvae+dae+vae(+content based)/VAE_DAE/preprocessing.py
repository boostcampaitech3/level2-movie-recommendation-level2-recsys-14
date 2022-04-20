import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from dataset import *
from utils import *
import re
import json
import gensim
import sister

## 각종 파라미터 세팅
parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

parser.add_argument('--data', type=str, default='data/train/', help='Movielens dataset location')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000, help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
args = parser.parse_args([])

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)

#만약 GPU가 사용가능한 환경이라면 GPU를 사용
if torch.cuda.is_available():
    args.cuda = True

device = torch.device("cuda" if args.cuda else "cpu")


print("Load and Preprocess Movielens dataset")
# Load Data
base_dir = '/opt/ml/input/'
DATA_DIR = base_dir + args.data
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
print("raw_data : Done!!")

def item_genre_emb_sum(x):
    total.append(np.mean(genres.loc[genres['item'] == x, 'emb']))

def title_numerize(tp, show2id):
    sid = tp['item'].apply(lambda x: show2id[x])
    return sid

# Filter Data
raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

# Shuffle User Indices
unique_uid = user_activity.index
print("(BEFORE) unique_uid:",unique_uid)
np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size) # 해당 숫자까지의 인덱스를 무작위로 섞은 것을 arr로 반환
unique_uid = unique_uid[idx_perm]
print("(AFTER) unique_uid:",unique_uid) # 무작위로 item을 섞음

n_users = unique_uid.size #31360
n_heldout_users = 3000

# Split Train/Validation/Test User Indices
tr_users = unique_uid[:(n_users - n_heldout_users * 2)] # [:25360]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)] # [25360 : 28360]
te_users = unique_uid[(n_users - n_heldout_users):] # [28360:]

#주의: 데이터의 수가 아닌 사용자의 수입니다!
print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
print("검증 데이터에 사용될 사용자 수:", len(vd_users))
print("테스트 데이터에 사용될 사용자 수:", len(te_users))

##훈련 데이터에 해당하는 아이템들
#Train에는 전체 데이터를 사용합니다.
train_plays = raw_data.loc[raw_data['user'].isin(tr_users)] # 25360명의 user에 대한 df

##아이템 ID
unique_sid = pd.unique(train_plays['item']) # 25360명이 평가한 unique item

show2id = dict((int(sid), int(i)) for (i, sid) in enumerate(unique_sid)) # item2idx dict
profile2id = dict((int(pid), int(i)) for (i, pid) in enumerate(unique_uid)) # user2idx dict

with open('show2id.json', 'w', encoding="utf-8") as f:
    json.dump(show2id, f, ensure_ascii=False, indent="\t")
    
with open('profile2id.json', 'w', encoding="utf-8") as f:
    json.dump(profile2id, f, ensure_ascii=False, indent="\t")


pro_dir = os.path.join(DATA_DIR, 'pro_sg')

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

#Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
print('Data Split Start!')
vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

train_data = numerize(train_plays, profile2id, show2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te, profile2id, show2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr, profile2id, show2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te, profile2id, show2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

print("Data Split Done!")

# from gensim import models
w2v = gensim.models.KeyedVectors.load_word2vec_format('/opt/ml/input/code/GoogleNews-vectors-negative300.bin.gz', binary=True)
genres = pd.read_csv('/opt/ml/input/data/train/genres.tsv', delimiter='\t')
genres['item'] = genres['item'].apply(lambda x : show2id[x])
genre_emb = pd.DataFrame(genres['genre'].value_counts().index, columns=['genre'])

emb_li = []
for x in genre_emb['genre']:
    if x == 'Sci-Fi':
        emb_li.append(w2v['Science_Fiction'])
    elif x == 'Film-Noir':
        emb_li.append(w2v['Film_Noir'])
    else:
        emb_li.append(w2v[x])

gen_emb = pd.concat([genre_emb, pd.DataFrame(emb_li)], axis=1)
gen_emb.set_index('genre', drop=True, inplace=True)

gen2emb = {x : gen_emb.loc[x].values for x in gen_emb.index}
genres['emb'] = genres['genre'].apply(lambda x : gen2emb[x])

total = []
temp = pd.DataFrame(list(i for i in range(0, max(genres.item)+1)), columns=['item'])
temp['item'].apply(lambda x : item_genre_emb_sum(x))
item_genre_emb = pd.DataFrame(total)
item_genre_emb = item_genre_emb.T
print('item_genre_emb.shape :', item_genre_emb.shape)
item_genre_emb.to_csv("item_genre_emb.csv", index=False)
print('item_genre_emb Done!!')

sentence_embedding = sister.MeanEmbedding(lang="en")
title = pd.read_csv("/opt/ml/input/data/train/titles.tsv", delimiter='\t')

title['item'] = title_numerize(title, show2id)

new_title = []

for item, text in title.values:
    new_text = ''
    if re.search(r'\([0-9]{4}\)', text[-6:]):
        new_text = text[:-6]
    else:
        new_text = text[:text.rfind('(')]

    # new_text = re.compile("\W+").sub(" ", new_text)
    # filtered_tokens = [token for token in word_tokenize(new_text) if token.lower() not in stops]
    # new_title.append([item, ' '.join(filtered_tokens)])
    new_title.append([item, new_text])

new_title_df = pd.DataFrame(new_title, columns=['item', 'title'])

emb_title_dict = {}
for item, title in new_title_df.values:
    emb_title_dict[item] = sentence_embedding(title)

emb_title_df = pd.DataFrame(list(emb_title_dict.items()), columns=['item', 'title'])
emb_title_df2 = emb_title_df.sort_values(by=['item'])

total_emb_title = []

for text in emb_title_df2['title'].values:
    text = text.tolist()
    total_emb_title.append(text)

item_title_emb = pd.DataFrame(total_emb_title)
item_title_emb = item_title_emb.T
print('item_title_emb.shape :', item_title_emb.shape)
item_genre_emb.to_csv("item_title_emb.csv", index=False)
print('item_title_emb Done!!')