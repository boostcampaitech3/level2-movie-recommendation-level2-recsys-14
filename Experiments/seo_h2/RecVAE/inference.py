import argparse
from cgi import test
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import sparse
import os
from dataset import *
from utils import *
import json
from tqdm import tqdm
from models import *
from utils import recall

import bottleneck as bn

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)

class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)
## 각종 파라미터 세팅
parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

parser.add_argument('--data', type=str, default='data/train/', help='Movielens dataset location')
parser.add_argument('--lr', type=float, default=0.0007, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000, help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--record', type=str, default= 'te')
args = parser.parse_args([])

## 배치사이즈 포함

### 데이터 준비
print("Inference Start!!")
# Load Data
base_dir = '/opt/ml/input/'
DATA_DIR = base_dir + args.data
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

with open('show2id.json', 'r', encoding="utf-8") as f:
    show2id = json.load(f)

with open('profile2id.json', 'r', encoding="utf-8") as f:
    profile2id = json.load(f)
loader = DataLoader(base_dir + args.data)
n_items = loader.load_n_items()

if args.record == 'test':
    tp_tr=pd.read_csv(os.path.join(DATA_DIR, '/opt/ml/input/data/train/pro_sg/test_tr.csv'), header=0)
    tp_te= pd.read_csv(os.path.join(DATA_DIR, '/opt/ml/input/data/train/pro_sg/test_te.csv'), header=0)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    n_users=end_idx - start_idx + 1

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data = sparse.csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64', shape=(n_users, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64', shape=(n_users, n_items))

else:
    infer_df = numerize_for_infer(raw_data, profile2id, show2id)

    n_users = infer_df['uid'].max() + 1

    rows, cols = infer_df['uid'], infer_df['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64',
                                    shape=(n_users, n_items))

device = torch.device("cuda")
model = torch.load('recvae7.pt').to(device)
# model_dae = torch.load('dae_model.pt').to(device)
# model_vae = torch.load('vae_model.pt').to(device)


model.eval()
# model_dae.eval()
# model_vae.eval()

# with open("update_count_vae.txt","r", encoding='utf-8') as f:
#         update_count_vae = int(f.readline())

pred_list = None
recon_list= None
seq=0
with torch.no_grad():
    for batch in tqdm(generate(batch_size=args.batch_size, device=device, data_in=data, shuffle=False)):
        # if args.total_anneal_steps > 0:
        #     anneal = min(args.anneal_cap, 1. * update_count_vae / args.total_anneal_steps)
        # else:
        #     anneal = args.anneal_cap
        seq+=1
        data_tensor =  batch.get_ratings_to_dev()

        recon_batch = model(data_tensor, calculate_loss=False)
        # recon_batch_dae = model_dae(data_tensor)
        # recon_batch_vae, mu, logvar = model_vae(data_tensor)

        recon_batch = recon_batch.cpu().numpy()
        # recon_batch_dae = recon_batch_dae.cpu().numpy()
        # recon_batch_vae = recon_batch_vae.cpu().numpy()

        # recon_batch += recon_batch_dae + recon_batch_vae

        recon_batch[batch._data_in[batch.get_idx()].nonzero()] = -np.inf

        ##Recall
        batch_users = recon_batch.shape[0]
        if batch.get_idx()[0] == 0:
            recon_list= recon_batch
        else:
            recon_list= np.append(recon_list, recon_batch, axis=0)
  
        # 배치 마지막에 수행
        if seq==(n_users//args.batch_size+1):
            # 감독선호도 추가
            pref_item0= np.load("/opt/ml/input/data/train/pref_item0.npy")
            pref_test0= np.load("/opt/ml/input/data/train/pref_test0.npy")

            # 장르선호도 추가
            pref_genre= np.load("/opt/ml/input/data/train/lst.npy")

            if args.record == 'test':
                recon_list*=pref_test0
            else:
                recon_list*=pref_item0
            recon_list+= pref_genre/100
            
            # 상위 10개 추출 
            pred_list = bn.argpartition(-recon_list, 50, axis=1)[:, :50]


if args.record == 'test':
    print(np.mean(recall(recon_list, data_te, k=10)))

else:
    user2 = []
    item2 = []
    for i_idx, arr_10 in enumerate(pred_list):
        user2.extend([i_idx]*50)
        item2.extend(arr_10)

    u2 = pd.DataFrame(user2, columns=['user'])
    i2 = pd.DataFrame(item2, columns=['item'])
    all2 = pd.concat([u2, i2], axis=1)

    re_p2id = dict((int(v), int(k)) for k, v in profile2id.items())
    re_s2id = dict((int(v), int(k)) for k, v in show2id.items())

    ans2 = de_numerize(all2, re_p2id, re_s2id)
    ans2.columns = ['user', 'item']
    new_ans2 = ans2.sort_values('user')

    new_ans2.to_csv('recvae_dir_genre7_50.csv', index=False)
    print('*****output saved!*****')