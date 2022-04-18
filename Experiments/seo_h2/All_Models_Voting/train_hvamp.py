import argparse
import datetime
import math
import os
import time

import bottleneck as bn
import numpy as np
import torch
from torch.autograd import Variable

from dataset import load_dataset
from utils.optimizer import AdamNormGrad
from utils.utils import increment_path

# Training settings
parser = argparse.ArgumentParser(description='H+Vamp')

# arguments for optimization
parser.add_argument('--data', type=str, default='./data', help='Movielens data location')
parser.add_argument('--batch_size', type=int, default=200, metavar='BStrain',
                    help='input batch size for training (default: 200)')
parser.add_argument('--split_mode', type=int, default=1, help='data split mode')
parser.add_argument('--fold_size', type=int, default=3000, help='fold size for k-fold')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='BStest',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='E',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warm-up')
parser.add_argument('--max_beta', type=float, default=1., metavar='B',
                    help='maximum value of beta for training')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

# model: latent size, input_size, so on
parser.add_argument('--num_layers', type=int, default=1, metavar='NL',
                    help='number of layers')

parser.add_argument('--z1_size', type=int, default=200, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=200, metavar='M2',
                    help='latent size')
parser.add_argument('--hidden_size', type=int, default=600, metavar="H",
                    help='the width of hidden layers')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=1000, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='hvamp', metavar='MN',
                    help='model name: baseline, vamp, hvamp, hvamp1')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous, multinomial')

parser.add_argument('--gated', action='store_true', default=True,
                    help='use gating mechanism')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

# note
parser.add_argument('--note', type=str, default="none", metavar='NT',
                    help='additional note on the experiment')
parser.add_argument('--log_dir', type=str, default='./logs', help='Logs directory')
parser.add_argument('--save', type=str, default='model.pt', help='file name to save the final model')
parser.add_argument('--save_dir', type=str, default='./latest', help='directory to save the latest best model')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ======================================================================================================================
def train(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = args.max_beta #! Changed value of beta to variable max_beta, newly given in train_hvamp.py
    else:
        beta = args.max_beta * epoch / args.warmup
        if beta > args.max_beta:
            beta = args.max_beta
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.calculate_loss(x, beta, average=True)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.data.item()
        train_re += -RE.data.item()
        train_kl += KL.data.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl

# ======================================================================================================================
def evaluate(args, model, train_loader, data_loader, epoch, log_dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0

    ndcg_dist = torch.tensor([], dtype=torch.float)
    if mode == 'test':
        ndcg_20 = torch.tensor([], dtype=torch.float)
        ndcg_10 = torch.tensor([], dtype=torch.float)
        recall_50 = torch.tensor([], dtype=torch.float)
        recall_20 = torch.tensor([], dtype=torch.float)
        recall_10 = torch.tensor([], dtype=torch.float)
        recall_5 = torch.tensor([], dtype=torch.float)
        recall_1 = torch.tensor([], dtype=torch.float)

    # set model to evaluation mode
    model.eval()

    # Functions for Evaluation

    def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
        '''
        normalized discounted cumulative gain@k for binary relevance
        ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
        '''
        batch_users = X_pred.shape[0]
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                           idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
        # topk predicted score
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        # build the discount template
        tp = 1. / np.log2(np.arange(2, k + 2))
        tp = torch.tensor(tp, dtype=torch.float)  # ! in order to do operations with torch tensor

        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].cpu() * tp).sum(dim=1)
        IDCG = torch.tensor([(tp[:min(n, k)]).sum()
                             for n in (heldout_batch != 0).sum(dim=1)])
        return DCG / IDCG

    def Recall_at_k_batch(X_pred, heldout_batch, k=100):
        batch_users = X_pred.shape[0]

        idx = bn.argpartition(-X_pred, k, axis=1)
        X_pred_binary = np.zeros_like(X_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

        # X_true_binary = torch.tensor((heldout_batch > 0), dtype=torch.float)
        X_true_binary = (heldout_batch > 0).clone().detach()
        X_true_binary = X_true_binary.cpu().detach()


        # tmp = torch.tensor(np.logical_and(X_true_binary, X_pred_binary), dtype=torch.float).sum(dim=1)
        tmp = torch.tensor(np.logical_and(X_true_binary.numpy(), X_pred_binary), dtype=torch.float).sum(dim=1)
        recall = tmp / np.minimum(k, X_true_binary.cpu().detach().sum(dim=1))
        return recall

    # evaluate
    for batch_idx, (train, test) in enumerate(data_loader):
        if args.cuda:
            train, test = train.cuda(), test.cuda()
        train, test = Variable(train), Variable(test) #! volatile deprecated

        x = train

        with torch.no_grad():
            # calculate loss function
            loss, RE, KL = model.calculate_loss(x, average=True)

            evaluate_loss += loss.data.item()
            evaluate_re += -RE.data.item()
            evaluate_kl += KL.data.item()

            # Calculate NDCG & Recall
            pred_val = model.reconstruct_x(x).cpu().detach()
            # should be removed if not necessary
            pred_val = np.array(pred_val)
            x = np.array(x.cpu().detach())
            pred_val[x.nonzero()] = -np.inf

            ndcg_dist = torch.cat([ndcg_dist, NDCG_binary_at_k_batch(pred_val, test, k=100)])

            if mode == 'test':
                ndcg_20 = torch.cat([ndcg_20, NDCG_binary_at_k_batch(pred_val, test, k=20)])
                ndcg_10 = torch.cat([ndcg_10, NDCG_binary_at_k_batch(pred_val, test, k=10)])
                recall_50 = torch.cat([recall_50, Recall_at_k_batch(pred_val, test, k=50)])
                recall_20 = torch.cat([recall_20, Recall_at_k_batch(pred_val, test, k=20)])
                recall_10 = torch.cat([recall_10, Recall_at_k_batch(pred_val, test, k=10)])
                recall_5 = torch.cat([recall_5, Recall_at_k_batch(pred_val, test, k=5)])
                recall_1 = torch.cat([recall_1, Recall_at_k_batch(pred_val, test, k=1)])



    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size

    evaluate_ndcg = ndcg_dist.mean().data.item()

    if mode == 'test':
        eval_ndcg100 = "{:.5f}({:.4f})".format(evaluate_ndcg, ndcg_dist.std().data.item()/np.sqrt(len(ndcg_dist)))
        eval_ndcg20 = "{:.5f}({:.4f})".format(ndcg_20.mean().data.item(),ndcg_20.std().data.item()/np.sqrt(len(ndcg_20)))
        eval_ndcg10 = "{:.5f}({:.4f})".format(ndcg_10.mean().data.item(),ndcg_10.std().data.item()/np.sqrt(len(ndcg_10)))
        eval_recall50 = "{:.5f}({:.4f})".format(recall_50.mean().data.item(),recall_50.std().data.item()/np.sqrt(len(recall_50)))
        eval_recall20 = "{:.5f}({:.4f})".format(recall_20.mean().data.item(),recall_20.std().data.item()/np.sqrt(len(recall_20)))
        eval_recall10 = "{:.5f}({:.4f})".format(recall_10.mean().data.item(),recall_10.std().data.item()/np.sqrt(len(recall_10)))
        eval_recall5 = "{:.5f}({:.4f})".format(recall_5.mean().data.item(),recall_5.std().data.item()/np.sqrt(len(recall_5)))
        eval_recall1 = "{:.5f}({:.4f})".format(recall_1.mean().data.item(),recall_1.std().data.item()/np.sqrt(len(recall_1)))

    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, eval_ndcg100, \
               eval_ndcg20, eval_ndcg10, eval_recall50, eval_recall20, eval_recall10, eval_recall5, eval_recall1
    else:
        return evaluate_loss, evaluate_re, evaluate_kl, evaluate_ndcg


# ======================================================================================================================
def experiment(args, train_loader, val_loader, test_loader, model, optimizer, log_dir):
    # SAVING
    # best_model = model
    best_ndcg = 0.
    e = 0
    last_epoch = 0

    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    val_ndcg_history = []

    time_history = []

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                             optimizer)

        val_loss_epoch, val_re_epoch, val_kl_epoch, val_ndcg_epoch = evaluate(args, model, train_loader, val_loader, epoch, log_dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch), val_ndcg_history.append(val_ndcg_epoch)
        time_history.append(time_elapsed)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
              '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f}, NDCG: {:.5f})\n'
              '--> Early stopping: {}/{} (BEST: {:.5f})\n'.format(
            epoch, args.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch,
            val_loss_epoch, val_re_epoch, val_kl_epoch, val_ndcg_epoch,
            e, args.early_stopping_epochs, best_ndcg
        ))

        # early-stopping
        last_epoch = epoch
        if val_ndcg_epoch > best_ndcg:
            e = 0
            best_ndcg = val_ndcg_epoch
            # best_model = model
            with open(os.path.join(log_dir, f'best_hvamp{args.split_mode}_' + args.save), 'wb') as f:
                torch.save(model, f)
                print('model saved!!')
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_loss_epoch):
            break

    # FINAL EVALUATION
    with open(os.path.join(log_dir, f'best_hvamp{args.split_mode}_' + args.save), 'rb') as f:
        best_model = torch.load(f)

    with open(os.path.join(args.save_dir, f'best_hvamp{args.split_mode}_' + args.save), 'wb') as f:
        torch.save(best_model, f)

    test_loss, test_re, test_kl, test_ndcg, \
    eval_ndcg20, eval_ndcg10, eval_recall50, eval_recall20, \
    eval_recall10, eval_recall5, eval_recall1 = evaluate(args, best_model, train_loader, test_loader, 9999, log_dir, mode='test')

    print("NOTE: " + args.note)
    print('FINAL EVALUATION ON TEST SET\n'
          '- BEST VALIDATION NDCG: {:.5f} ({:} epochs) -\n'
          'NDCG@100: {:}  |  Loss: {:.2f}\n'
          'NDCG@20: {:}   |  RE: {:.2f}\n'
          'NDCG@10: {:}   |  KL: {:.2f}\n'
          'Recall@50: {:} |  Recall@5: {:}\n'
          'Recall@20: {:} |  Recall@1: {:}\n'
          'Recall@10: {:}'.format(
        best_ndcg, last_epoch,
        test_ndcg, test_loss,
        eval_ndcg20, test_re,
        eval_ndcg10, test_kl,
        eval_recall50, eval_recall5,
        eval_recall20, eval_recall1,
        eval_recall10
    ))
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}  #! Changed num_workers: 1->0 because of error

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    # DIRECTORY FOR SAVING
    log_dir_name = str(datetime.datetime.now())[0:10] + f'_hvamp{args.split_mode}'
    log_dir = os.path.join(args.log_dir, log_dir_name)
    log_dir = increment_path(log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    assert os.path.exists(args.data), "Preprocessed files do not exist."
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'gated_vae':
        from models import GatedVAE as VAE
    elif args.model_name == 'hvamp':
        from models import Hvamp as VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    if args.cuda:
        model.cuda()

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # ======================================================================================================================
    print(args)

    # ======================================================================================================================
    print('perform experiment')
    experiment(args, train_loader, val_loader, test_loader, model, optimizer, log_dir)
    # ======================================================================================================================


if __name__ == "__main__":
    run(args, kwargs)


