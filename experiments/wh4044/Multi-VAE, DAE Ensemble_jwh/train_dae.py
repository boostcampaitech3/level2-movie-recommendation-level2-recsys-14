import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import sparse
import os
import adabound

import wandb
from metrics import *
from models import *
from utils import *
from dataset import *


## 각종 파라미터 세팅
parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

parser.add_argument('--data', type=str, default='data/train/', help='Movielens dataset location')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000, help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
args = parser.parse_args([])

# Set the random seed manually for reproductibility.
# torch.manual_seed(args.seed)

#만약 GPU가 사용가능한 환경이라면 GPU를 사용
if torch.cuda.is_available():
    args.cuda = True

device = torch.device("cuda" if args.cuda else "cpu")

def train(model, criterion, optimizer, is_VAE = False):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        # print('batch_idx :', batch_idx)
        end_idx = min(start_idx + args.batch_size, N)
        
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)
        
        #-------------------------------------------------------------
        # print('naive_sparse2tensor(data) :\n', data)
        # print(data.size())
        # return
        #-------------------------------------------------------------
        optimizer.zero_grad()

        if is_VAE:
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            #------------------------------------------------------
#             print('recon_batch :\n', recon_batch)
#             print(recon_batch.size())
            
#             print('mu :\n', mu)
#             print(mu.size())
            
#             print('logvar :\n', logvar)
#             print(logvar.size())
#             return
            #------------------------------------------------------
            loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
            recon_batch = model(data)
            loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

#         if batch_idx % args.log_interval == 0 and batch_idx > 0:
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
#                     'loss {:4.2f}'.format(
#                         epoch, batch_idx, len(range(0, N, args.batch_size)),
#                         elapsed * 1000 / args.log_interval,
#                         train_loss / args.log_interval))
            

#             start_time = time.time()
#             train_loss = 0.0


def evaluate(model, criterion, data_tr, data_te, is_VAE=False):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r10_list = []
    r20_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)
            if is_VAE :

                if args.total_anneal_steps > 0:
                    anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
                else:
                    anneal = args.anneal_cap

                recon_batch, mu, logvar = model(data_tensor)

                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
                recon_batch = model(data_tensor)
                loss = criterion(recon_batch, data_tensor)

            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)

            n100_list.append(n100)
            r10_list.append(r10)
            r20_list.append(r20)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r10_list = np.concatenate(r10_list)
    r20_list = np.concatenate(r20_list)

    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list)

###############################################################################
# Load data
###############################################################################
base_dir = '/opt/ml/input/'
loader = DataLoader(base_dir + args.data)

n_items = loader.load_n_items() # 6807
train_data = loader.load_data('train') # csr_matrix
vad_data_tr, vad_data_te = loader.load_data('validation')
test_data_tr, test_data_te = loader.load_data('test')

N = train_data.shape[0] # 25360
idxlist = list(range(N))

###############################################################################
# Build the model
###############################################################################

# p_dims = [200, 1200, 3000, n_items] # [200, 600, 6807]
p_dims = [200, 3000, n_items] # [200, 600, 6807]
item_genre_emb = pd.read_csv('item_genre_emb.csv')
model = MultiDAE(p_dims, genre_emb = item_genre_emb).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
# optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
criterion = loss_function_dae

###############################################################################
# Training code
###############################################################################

# best_n100 = -np.inf
best_r10 = -np.inf
update_count = 0
early_stopping = 40
stopping_cnt = 0

# -- wandb initialize with configuration
wandb.init(config={"model":'Multi-DAE',
                "batch_size": args.batch_size,
                "lr"        : args.lr,
                "epochs"    : args.epochs,
                })



for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, criterion, optimizer, is_VAE=False)
    
    
    val_loss, n100, r10, r20 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE=False)
    print('-' * 89)
    print('| end of epoch {:3d}/{:3d} | time: {:4.2f}s | valid loss {:4.4f} | '
            'n100 {:5.4f} | r10 {:5.4f} | r20 {:5.4f}'.format(
                epoch, args.epochs, time.time() - epoch_start_time, val_loss,
                n100, r10, r20))
    print('-' * 89)

    n_iter = epoch * len(range(0, N, args.batch_size))

    wandb.log({
        "dae_val loss": val_loss,
        "dae_n100" : n100,
        "dae_r10" : r10,
        "dae_r20" : r20})


    # Save the model if the n100 is the best we've seen so far.
    # if n100 > best_n100:
    #     with open(args.save, 'wb') as f:
    #         torch.save(model, f)
    #     best_n100 = n100

    if r10 > best_r10:
        with open('dae_' + args.save, 'wb') as f:
            torch.save(model, f)
            print(f"Best model saved! r@10 : {r10:.4f}")
        best_r10 = r10
        stopping_cnt = 0
    else:
        print(f'Stopping Count : {stopping_cnt} / {early_stopping}')
        stopping_cnt += 1

    if stopping_cnt > early_stopping:
        print('*****Early Stopping*****')
        break

# Load the best saved model.
with open('dae_' + args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, n100, r10, r20 = evaluate(model, criterion, test_data_tr, test_data_te, is_VAE=False)
print('=' * 89)
print('| End of training | test loss {:4.4f} | n100 {:4.4f} | r10 {:4.4f} | '
        'r20 {:4.4f}'.format(test_loss, n100, r10, r20))
print('=' * 89)

with open("update_count_dae.txt", "w", encoding='utf-8') as f:
    f.write(str(update_count))