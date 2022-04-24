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
from models import *
import wandb

def change(x):
    x_ = x.detach().cpu().numpy()
    li = [5.0 if i >= 4.0 else 0.0 for i in x_]
    arr = np.array(li)
    return torch.tensor(arr)

print('load data.csv.....')
data = pd.read_csv('data.csv')
print('load data.csv Done!***')
n_data = len(data)
n_user = data['user'].nunique()
n_item = data['item'].nunique()
n_genre = data['genre'].nunique()
n_year = data['year'].nunique()
print("# of data : {}\n# of users : {}\n# of items : {}\n# of genres : {}\n# of years : {}".format(n_data, n_user, n_item, n_genre, n_year))

#6. feature matrix X, label tensor y 생성
user_col = torch.tensor(np.array(data['user']))
item_col = torch.tensor(np.array(data['item']))
genre_col = torch.tensor(np.array(data['genre']))
year_col = torch.tensor(np.array(data['year']))

n = n_user+n_item
offsets = [0, n_user, n, n + n_genre] # one-hot 인코딩 느낌을 내기 위해 각 원소에 원소의 개수만큼 더해준다.

for col, offset in zip([user_col, item_col, genre_col, year_col], offsets):
    col += offset

# [user_id, item_id, genre_id] 형태로 되도록 unsqueeze(1)을 취해줘서 cat.
X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1), year_col.unsqueeze(1)], dim=1)
y = torch.tensor(np.array(data['rating']))

# Custom Dataset
dataset = RatingDataset(X, y)
train_ratio = 0.8

# train:test = 8:2 비율로 random split
train_size = int(train_ratio * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Training
print("Training.......")
device = torch.device('cuda')
input_dims = [n_user, n_item, n_genre, n_year]
embedding_dim = 10

model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)

bce_loss = nn.BCELoss() # Binary Cross Entropy loss
mse = nn.MSELoss()
lr, num_epochs = 0.01, 10
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_ = []
min_loss = int(1e9)
early_stopping = 10
stopping_cnt = 0
loader_len = len(train_loader)

# # -- wandb initialize with configuration
# wandb.init(config={"batch_size": 1024,
#                 "lr"        : lr,
#                 "epochs"    : num_epochs,
#                 "criterion_name" : 'MSE'})


for e in tqdm(range(1, num_epochs+1)):
    model.train()

    now_loss = 0
    loader_cnt = 0
    train_correct = 0
    train_total = 0
    time_a = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        model.train()
        optimizer.zero_grad()
        output = model(x)

        loss = bce_loss(output, y.float())
        # loss = mse(output, y.float())
        loss.backward()
        optimizer.step()
        
        # result = change(output).to(device)
        result = torch.round(output)

        now_loss += loss.item()
        loader_cnt += 1
        train_total += y.size(0)
        train_correct += (result == y).sum().item()
        # print(result)
        # print(y)
        # break
        if loader_cnt % 1400 == 0:
            time_b = time.time()
            print(f"[{e}], loader : {loader_cnt} / {loader_len} [{time_b-time_a:.2f}sec]")
            time_a = time.time()
    # break
    now_loss /= loader_cnt
    
    # loss_.append(now_loss)
    
    if now_loss <= min_loss:
        min_loss = now_loss
        stopping_cnt = 0
        torch.save(model, "best_model.pt")
        print(f"epoch : {e}, best_model saved!*********************")
    else:
        stopping_cnt += 1
        
    if stopping_cnt > early_stopping:
        print(f"EARLY STOPPING")
        break
    
    print(f'epoch : {e}/{num_epochs}, Train ACC : {train_correct/train_total*100:.4f}%, now_loss : {now_loss:.4f} , min_loss : {min_loss:.4f}')
    print(f'stop : {stopping_cnt}/{early_stopping}')
    print("="*80)

    # wandb.log({
    #     "Train loss": now_loss,
    #     "Train acc" : train_correct/train_total*100})
else:
    print("****Training Done!!****")

# model = torch.load('best_model.pt')

# Testset Evaluation
print('\nAcc Evaluation*********\n')
correct_result_sum = 0
for x, y in tqdm(test_loader):
    x, y = x.to(device), y.to(device)
    model.eval()
    output = model(x)
    result = torch.round(output)
    # result = change(output).to(device)
    correct_result_sum += (result == y).sum().float()

acc = correct_result_sum/len(test_dataset)*100
print("Final Acc : {:.4f}%".format(acc.item()))