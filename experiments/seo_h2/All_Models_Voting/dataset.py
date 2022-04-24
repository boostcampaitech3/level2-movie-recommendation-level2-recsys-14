import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from scipy import sparse

"""
DataLoader for Multi-DAE and Multi-VAE
"""


class DataLoader:
    """
    Load Movielens dataset
    """

    def __init__(self, path, split_mode, fold_size=3000):
        self.path_dir = path
        self.split_mode = split_mode
        self.fold_size = fold_size
        assert os.path.exists(self.path_dir), "Preprocessed files do not exist."

        self.n_items = self.load_n_items()
        self.n_users = self.load_n_users()

    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        """
        아이템의 개수를 반환하는 함수.
        """
        unique_sid = list()
        with open(os.path.join(self.path_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def load_n_users(self):
        unique_uid = list()
        with open(os.path.join(self.path_dir, 'unique_uid.txt'), 'r') as f:
            for line in f:
                unique_uid.append(line.strip())
        n_users = len(unique_uid)
        return n_users

    def _load_train_data(self):
        """
        train 데이터를 csr_matrix로 반환해주는 함수
        """
        path = os.path.join(self.path_dir, f'train{self.split_mode}.csv')

        tp = pd.read_csv(path)
        # n_users = tp['uid'].max() + 1
        #
        # rows, cols = tp['uid'], tp['sid']
        # data = sparse.csr_matrix((np.ones_like(rows),
        #                           (rows, cols)), dtype='float64',
        #                          shape=(n_users, self.n_items))

        data = None

        if 1 <= self.split_mode <= 3:
            start_idx = tp['uid'].min()
            end_idx = tp['uid'].max()

            rows, cols = tp['uid'] - start_idx, tp['sid']

            data = sparse.csr_matrix((np.ones_like(rows),
                                      (rows, cols)), dtype='float64',
                                     shape=(end_idx - start_idx + 1, self.n_items))
        elif 4 <= self.split_mode <= 6:
            rows = pd.concat([tp[tp['uid'] < self.fold_size * (self.split_mode - 3)]['uid'],
                              tp[tp['uid'] >= self.fold_size * (self.split_mode - 1)]['uid'] - self.fold_size * 2])
            cols = tp['sid']
            data = sparse.csr_matrix((np.ones_like(rows),
                                      (rows, cols)), dtype='float64',
                                     shape=(self.n_users - self.fold_size * 2, self.n_items))

        return data

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.path_dir, '{}_tr{}.csv'.format(datatype, self.split_mode))
        te_path = os.path.join(self.path_dir, '{}_te{}.csv'.format(datatype, self.split_mode))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te


"""
DataLoader for RecVAE
"""


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


def generate_batch(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
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
        print("batch idx: ", idx)

        yield Batch(device, idx, data_in, data_out)


"""
DataLoader for H+vamp
"""


def load_dataset(args, **kwargs):
    unique_sid = list()
    with open(os.path.join(args.data, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    unique_uid = list()
    with open(os.path.join(args.data, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    n_users = len(unique_uid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        # n_users = tp['uid'].max() + 1
        #
        # rows, cols = tp['uid'], tp['sid']
        # data = sparse.csr_matrix((np.ones_like(rows),
        #                           (rows, cols)), dtype='float32',
        #                          shape=(n_users, n_items)).toarray()

        data = None

        if 1 <= args.split_mode <= 3:
            start_idx = tp['uid'].min()
            end_idx = tp['uid'].max()

            rows, cols = tp['uid'] - start_idx, tp['sid']

            data = sparse.csr_matrix((np.ones_like(rows),
                                      (rows, cols)), dtype='float32',
                                     shape=(end_idx - start_idx + 1, n_items)).toarray()
        elif 4 <= args.split_mode <= 6:
            rows = pd.concat([tp[tp['uid'] < args.fold_size * (args.split_mode - 3)]['uid'],
                              tp[tp['uid'] >= args.fold_size * (args.split_mode - 1)]['uid'] - args.fold_size * 2])
            cols = tp['sid']
            data = sparse.csr_matrix((np.ones_like(rows),
                                      (rows, cols)), dtype='float32',
                                     shape=(n_users - args.fold_size * 2, n_items)).toarray()

        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te

    # train, validation and test data
    x_train = load_train_data(os.path.join(args.data, f'train{args.split_mode}.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join(args.data, f'validation_tr{args.split_mode}.csv'),
                                         os.path.join(args.data, f'validation_te{args.split_mode}.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join(args.data, f'test_tr{args.split_mode}.csv'),
                                           os.path.join(args.data, f'test_te{args.split_mode}.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy(
            init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components)).float()

    return train_loader, val_loader, test_loader, args


