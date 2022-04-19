import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from args import parse_args
from dataloader import DataLoader
from models import MultiDAE, MultiVAE
from optimizer import get_optimizer
from criterions import get_criterion
from metric import NDCG_binary_at_k_batch, Recall_at_k_batch
import pandas as pd
import warnings
import pickle
import bottleneck as bn
warnings.filterwarnings(action='ignore')
import os

## 배치사이즈 포함
def numerize_for_infer(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


if __name__ == "__main__":
    args = parse_args(mode="train")
    
    pro_dir = os.path.join(args.data_dir, 'pro_sg')
    with open(os.path.join(pro_dir, "profile2id.pkl"), 'rb') as f:
        profile2id = pickle.load(f)

    with open(os.path.join(pro_dir, "show2id.pkl"), 'rb') as f:
        show2id = pickle.load(f)

    ### 데이터 준비    
    raw_data = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'), header=0)
    infer_df = numerize_for_infer(raw_data, profile2id, show2id)

    loader = DataLoader(args.data_dir)
    n_items = loader.load_n_items()

    n_users = infer_df['uid'].max() + 1

    rows, cols = infer_df['uid'], infer_df['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64',
                                    shape=(n_users, n_items))

    N = data.shape[0]
    idxlist = list(range(N))
    
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    criterion = get_criterion(args)

    model.eval()
    total_loss = 0.0
    e_idxlist = list(range(data.shape[0]))
    e_N = data.shape[0]
    pred_list = None

    print("inference 중")
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data_batch = data[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data_batch).to(args.device)
            if args.model == "multivae" or args.model == "multivae_item_emb":
                recon_batch, mu, logvar = model(data_tensor)
            elif args.model == "multidae" or args.model == "multidae_item_emb":
                 recon_batch = model(data_tensor)

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data_batch.nonzero()] = -np.inf
    
            ##Recall
            batch_users = recon_batch.shape[0]
            idx = bn.argpartition(-recon_batch, 10, axis=1)[:, :10]
            if start_idx == 0:
                pred_list = idx
            else:
                pred_list = np.append(pred_list, idx, axis=0)

    
    print(pred_list.shape)
    ## sample_submission에 맞게끔 바꾸기
    user2 = []
    item2 = []
    for i_idx, arr_10 in enumerate(pred_list):
        user2.extend([i_idx]*10)
        item2.extend(arr_10)

    u2 = pd.DataFrame(user2, columns=['user'])
    i2 = pd.DataFrame(item2, columns=['item'])
    all2 = pd.concat([u2, i2], axis=1)

    re_p2id = dict((v, k) for k, v in profile2id.items())
    re_s2id = dict((v, k) for k, v in show2id.items())

    def de_numerize(tp, re_p2id, re_s2id):
        uid2 = tp['user'].apply(lambda x: re_p2id[x])
        sid2 = tp['item'].apply(lambda x: re_s2id[x])
        return pd.DataFrame(data={'uid': uid2, 'sid': sid2}, columns=['uid', 'sid'])

    ans2 = de_numerize(all2, re_p2id, re_s2id)
    ans2.columns = ['user', 'item']
    new_ans2 = ans2.sort_values('user')
    new_ans2.to_csv('submit.csv')
    print(new_ans2)

    print("inference 완료!")
    ### 확인용
    #submit_data = pd.read_csv('/content/data/eval/sample_submission.csv', sep='\t')
    #sum(new_ans2.user.values == submit_data.user.values)