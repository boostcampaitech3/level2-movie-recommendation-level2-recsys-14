import csv
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)