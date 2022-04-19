import argparse
import time
import bottleneck as bn
import numpy as np
import pandas as pd
from scipy import sparse
from args import parse_args
from models import EASE
import warnings
warnings.filterwarnings(action='ignore')


def runs(data, num_k):
    args = parse_args(mode="train")
    
    # Load data
    for_check = pd.read_csv('check_te.csv')
    for_train = pd.read_csv('check_tr.csv')

    train_pivot_table = for_train.pivot_table(index = ["uid"], columns = ["sid"],values = "rating")
    check_pivot_table = for_check.pivot_table(index = ["uid"], columns = ["sid"],values = "rating")

    X = train_pivot_table.to_numpy()
    X = np.nan_to_num(X)

    pivot_table = data.pivot_table(index = ["uid"], columns = ["sid"],values = "rating")
    X_ori = pivot_table.to_numpy()
    X_ori = np.nan_to_num(X_ori)

    check_x = check_pivot_table.to_numpy()
    check_x = np.nan_to_num(check_x)


    ease = EASE(300)
    ease.train(X_ori)

    # 비교 데이터 적용
    result = ease.forward(X[:, :])

    idx = bn.argpartition(-result, num_k, axis=1)
    X_pred_binary = np.zeros_like(result, dtype=bool)
    X_pred_binary[np.arange(result.shape[0])[:, np.newaxis], idx[:, :num_k]] = True
    X_true_binary = check_x > 0

    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
            np.float32)
    recall = tmp / np.minimum(num_k, X_true_binary.sum(axis=1))
    print('recall 결과: ', np.mean(recall))

    return ease, X, check_x

