import os
import pandas as pd
from scipy import sparse
import numpy as np
from args import parse_args
import pickle

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    '''
    특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상)
    데이터만을 추출할 때 사용하는 함수
    '''
        
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

    
def split_train_test_proportion(data, test_prop=0.2):
    '''
    훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
    100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지 확인하기 위함입니다.
    '''
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
        
    for _, group in data_grouped_by_user:
        n_items_u = len(group)
            
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
            
        else:
            tr_list.append(group)
        
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def process():
    args = parse_args(mode="train")

    # Filter Data
    min_uc = args.min_uc
    min_sc = args.min_sc    
    DATA_DIR = args.data_dir
    n_heldout_users = args.n_heldout

    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
    raw_data['rating'] = [1] * len(raw_data)
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc, min_sc)
    print(raw_data)
    print(min_uc)
    print(min_sc)
    print("5번 이상의 리뷰가 있는 유저들로만 구성된 데이터\n",raw_data)

    print("유저별 리뷰수\n",user_activity)
    print("아이템별 리뷰수\n",item_popularity)

    # Shuffle User Indices
    unique_uid = user_activity.index
    print("(BEFORE) unique_uid:",unique_uid)
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    unique_sid = pd.unique(raw_data['item'])
    print("(AFTER) unique_uid:",unique_uid)

    n_users = unique_uid.size #31360

    #Train에는 전체 데이터를 사용합니다.
    train_plays = raw_data.loc[raw_data['user']]
    ##아이템 ID
    unique_sid = pd.unique(train_plays['item'])
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    data = numerize(raw_data, profile2id, show2id)

    for_train, for_check = split_train_test_proportion(data)

    for_check['rating'] = 1
    for_train['rating'] = 1

    for_train.to_csv('check_tr.csv')
    for_check.to_csv('check_te.csv')

    
    pro_dir = os.path.join('pro_sg')
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    with open(os.path.join(pro_dir, "show2id.pkl"), "wb") as tf:
        pickle.dump(show2id,tf)
    with open(os.path.join(pro_dir, "profile2id.pkl"), "wb") as tf:
        pickle.dump(profile2id,tf)

    print("Done!")
    return data

     
