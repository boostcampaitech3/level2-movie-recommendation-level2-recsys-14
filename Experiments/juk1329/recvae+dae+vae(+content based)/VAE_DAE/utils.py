import os
import pandas as pd
from scipy import sparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

def get_count(tp, id):
    '''
    tp -> DataFrame
    id -> Feature of DataFrame
    '''
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count

# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상) 
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item') # tp DF의 'item'에 대한 count
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

#훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
#100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
#확인하기 위함입니다.
def split_train_test_proportion(data, test_prop=0.2):
    '''
    data -> DataFrame
    
    train과 test를 8:2 비율로 나눠주는 함수.
    '''
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
    
    for _, group in data_grouped_by_user:
        n_items_u = len(group)
        
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool') # 'False'가 n_items_u개 만큼 채워진 array
            
            # n_items_u개 중에서 20%의 인덱스를 랜덤으로 뽑아서 해당 인덱스를 'True'로 바꿈
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
                    
            tr_list.append(group[np.logical_not(idx)]) # 'False'인 것을 tr_list에 추가
            te_list.append(group[idx]) # 'True'인 것을 te_list에 추가
        
        else:
            tr_list.append(group)
    
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    '''
    tp -> DataFrame
    profile2id, show2id -> dict()
    
    user, item을 reindexing한 df 반환.
    '''
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def numerize_for_infer(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[str(x)])
    sid = tp['item'].apply(lambda x: show2id[str(x)])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def de_numerize(tp, re_p2id, re_s2id):
    uid2 = tp['user'].apply(lambda x: re_p2id[x])
    sid2 = tp['item'].apply(lambda x: re_s2id[x])
    return pd.DataFrame(data={'uid': uid2, 'sid': sid2}, columns=['uid', 'sid'])

