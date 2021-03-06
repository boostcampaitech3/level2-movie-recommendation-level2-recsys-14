import pandas as pd
import os
import bottleneck as bn
from scipy import sparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    tp = pd.read_csv(csv_file)
    
    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users, global_indexing=False):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    if global_indexing:
        start_idx = 0
        end_idx = len(unique_uid) - 1
    else:
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def get_data(dataset, global_indexing=False):
    unique_sid = list()
    with open(os.path.join(dataset, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    # unique_uid = list()
    # with open(os.path.join(dataset, 'unique_uid.txt'), 'r') as f:
    #     for line in f:
    #         unique_uid.append(line.strip())
            
    n_items = len(unique_sid)
    n_users = 31360
    
    train_data = load_train_data(os.path.join(dataset, 'train6.csv'), n_items, n_users, global_indexing=global_indexing)
    # add_train=load_train_data(os.path.join(datasest, 'add_train.csv'), n_items, n_users, global_indexing=global_indexing)

    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(dataset, 'validation_tr.csv'),
                                               os.path.join(dataset, 'validation_te.csv'),
                                               n_items, n_users, 
                                               global_indexing=global_indexing)

    test_data_tr, test_data_te = load_tr_te_data(os.path.join(dataset, 'test_tr.csv'),
                                                 os.path.join(dataset, 'test_te.csv'),
                                                 n_items, n_users, 
                                                 global_indexing=global_indexing)
    
    data = train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
    data = (x.astype('float32') for x in data)
    
    return data


def ndcg(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def recall(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def get_count(tp, id):
    '''
    tp -> DataFrame
    id -> Feature of DataFrame
    '''
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count

# ????????? ?????? ????????? ????????? ????????????(???????????? ?????? min_uc ??????, ???????????? ?????? min_sc??????) 
# ??????????????? ????????? ??? ???????????? ???????????????.
# ?????? ????????????????????? ??????????????? ??????????????? ???????????? ?????????.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item') # tp DF??? 'item'??? ?????? count
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

#????????? ????????? ????????? ????????? ???????????? ???????????? ???????????????.
#100?????? ????????? ?????????, ????????? test_prop ?????? ????????? ????????????, ????????? ????????? ????????? ??? ????????????
#???????????? ???????????????.
def split_train_test_proportion(data, test_prop=0.2):
    '''
    data -> DataFrame
    
    train??? test??? 8:2 ????????? ???????????? ??????.
    '''
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
    
    for _, group in data_grouped_by_user:
        n_items_u = len(group)
        
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool') # 'False'??? n_items_u??? ?????? ????????? array
            
            # n_items_u??? ????????? 20%??? ???????????? ???????????? ????????? ?????? ???????????? 'True'??? ??????
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
                    
            tr_list.append(group[np.logical_not(idx)]) # 'False'??? ?????? tr_list??? ??????
            te_list.append(group[idx]) # 'True'??? ?????? te_list??? ??????
        
        else:
            tr_list.append(group)
    
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    '''
    tp -> DataFrame
    profile2id, show2id -> dict()
    
    user, item??? reindexing??? df ??????.
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