import csv
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_genre = Sparse Features
        # print("total_input_dim :", total_input_dim)
        # total_input_dim : 38185

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,))) # w_0
        self.fc = nn.Embedding(total_input_dim, 1) # w_i * x_i, Linear를 써도 되는 듯..!?
        # self.fc.shape = (batch_size, 3, 1)
        
        self.embedding = nn.Embedding(total_input_dim, embedding_dim) # Sparse Features => embedding_dim(=10)
        self.embedding_dim = len(input_dims) * embedding_dim # 3개의 특성의 각 임베딩 차원(=10)을 연결시켜준다. MLP를 위해! ==> 30으로
        # print("self.embedding_dim :", self.embedding_dim)

        mlp_layers = []
        for i, dim in enumerate(mlp_dims): # [30, 20, 10]
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:                
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim))
            mlp_layers.append(nn.ReLU(True))                     
            mlp_layers.append(nn.Dropout(drop_rate))
            
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1)) 
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input) = (1024, 3)
        embed_x = self.embedding(x) # (batch_size, ,3, embedding_dim = 10)
        # embed_x.shape : torch.Size([1024, 3, 10])

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        
        embed_x = self.embedding(x)
        # print("embed_x.shape :", embed_x.size())
        # embed_x.shape : torch.Size([1024, 3, 10])
        
        inputs = embed_x.view(-1, self.embedding_dim)
        # print("inputs.shape :", inputs.size())
        # inputs.shape : torch.Size([1024, 30])
        
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        # print('x.shape :', x.size())
        # x.shape : torch.Size([1024, 3])
        
        # embed_x = self.embedding(x) # 이건 왜 한거지!?
        # print('embed_x :\n', embed_x)
        
        #fm component
        fm_y = self.fm(x).squeeze(1)
        
        #deep component
        mlp_y = self.mlp(x).squeeze(1)
        # print((fm_y + mlp_y)/100)
        y = torch.sigmoid((fm_y + mlp_y)/1000)
        return y