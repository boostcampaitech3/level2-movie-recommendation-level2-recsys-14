import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from dataset import *
from utils import *
import json
from tqdm import tqdm
from models import loss_function_vae, loss_function_dae
import bottleneck as bn


def numerize_for_infer(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


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

## 배치사이즈 포함

### 데이터 준비
print("Inference Start!!")
# Load Data
base_dir = '/opt/ml/input/'
DATA_DIR = base_dir + args.data
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

with open('show2id.json', 'r', encoding="utf-8") as f:
    show2id = json.load(f)

with open('profile2id.json', 'r', encoding="utf-8") as f:
    profile2id = json.load(f)

infer_df = numerize_for_infer(raw_data, profile2id, show2id)

loader = DataLoader(base_dir + args.data)
n_items = loader.load_n_items()

n_users = infer_df['uid'].max() + 1

rows, cols = infer_df['uid'], infer_df['sid']
data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))

N = data.shape[0]
idxlist = list(range(N))

device = torch.device("cuda")
model_dae = torch.load('dae_model.pt').to(device)
model_vae = torch.load('vae_model.pt').to(device)

model_dae.eval()
model_vae.eval()
total_dae_loss = 0.0
total_vae_loss = 0.0

e_idxlist = list(range(data.shape[0]))
e_N = data.shape[0]
pred_list = None
criterion_vae = loss_function_vae
criterion_dae = loss_function_dae

with open("update_count_dae.txt","r", encoding='utf-8') as f:
        update_count_dae = int(f.readline())
with open("update_count_vae.txt","r", encoding='utf-8') as f:
        update_count_vae = int(f.readline())

# print('update_counts :', update_count)

with torch.no_grad():
    for start_idx in tqdm(range(0, e_N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data_batch = data[e_idxlist[start_idx:end_idx]]

        data_tensor = naive_sparse2tensor(data_batch).to(device)
        data_tensor2 = naive_sparse2tensor(data_batch).to(device)
      
        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 1. * update_count_vae / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap
        # print("data_tensor.device :", data_tensor.device)
        recon_batch_vae, mu, logvar = model_vae(data_tensor)
        recon_batch_dae = model_dae(data_tensor)

        loss_vae = criterion_vae(recon_batch_vae, data_tensor2, mu, logvar, anneal)
        loss_dae = criterion_dae(recon_batch_dae, data_tensor)
        
        total_vae_loss += loss_vae.item()
        total_dae_loss += loss_dae.item()

        # Exclude examples from training set
        recon_batch_vae = recon_batch_vae.cpu().numpy()
        recon_batch_dae = recon_batch_dae.cpu().numpy()

        recon_batch_total = recon_batch_vae + recon_batch_dae
                
        recon_batch_total[data_batch.nonzero()] = -np.inf
  
        ##Recall
        batch_users = recon_batch_total.shape[0]
        idx = bn.argpartition(-recon_batch_total, 10, axis=1)[:, :10]
        if start_idx == 0:
            pred_list = idx
        else:
            pred_list = np.append(pred_list, idx, axis=0)

# print(pred_list.shape)
## sample_submission에 맞게끔 바꾸기
user2 = []
item2 = []
for i_idx, arr_10 in enumerate(pred_list):
    user2.extend([i_idx]*10)
    item2.extend(arr_10)

u2 = pd.DataFrame(user2, columns=['user'])
i2 = pd.DataFrame(item2, columns=['item'])
all2 = pd.concat([u2, i2], axis=1)

re_p2id = dict((int(v), int(k)) for k, v in profile2id.items())
re_s2id = dict((int(v), int(k)) for k, v in show2id.items())

ans2 = de_numerize(all2, re_p2id, re_s2id)
ans2.columns = ['user', 'item']
new_ans2 = ans2.sort_values('user')

new_ans2.to_csv('output.csv', index=False)
print('*****output saved!*****')