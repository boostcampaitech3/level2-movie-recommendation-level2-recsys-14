import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from args import parse_args
from dataloader import DataLoader
from models import MultiDAE, MultiVAE, MultiDAE_item_emb, MultiVAE_item_emb
from optimizer import get_optimizer
from criterions import get_criterion
from metric import NDCG_binary_at_k_batch, Recall_at_k_batch
import warnings
warnings.filterwarnings(action='ignore')

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train(model, criterion, optimizer, is_VAE):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(args.device)
        optimizer.zero_grad()

        if is_VAE:
          if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
          else:
              anneal = args.anneal_cap

          optimizer.zero_grad()
          recon_batch, mu, logvar = model(data)
          
          loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
          recon_batch = model(data)
          loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            

            start_time = time.time()
            train_loss = 0.0


def evaluate(model, criterion, data_tr, data_te, is_VAE):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r10_list= []
    r20_list = []
    r50_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(args.device)
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
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r10_list.append(r10)
            r50_list.append(r50)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r10_list = np.concatenate(r10_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)


def get_model(args, p_dims):
    """
    Load model and move tensors to a given devices.
    """
    p_dims = [200, 600, n_items]

    if args.model == "multidae":
        model = MultiDAE(p_dims).to(args.device)
    if args.model == "multivae":
        model = MultiVAE(p_dims).to(args.device)
    if args.model == "multidae_item_emb":
        model = MultiDAE_item_emb(p_dims).to(args.device)
    if args.model == "multivae_item_emb":
        model = MultiVAE_item_emb(p_dims).to(args.device)
    return model


if __name__ == "__main__":
    args = parse_args(mode="train")
    
    # Load data
    loader = DataLoader(args.data_dir)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')

    N = train_data.shape[0]
    idxlist = list(range(N))

    # Build the model
    p_dims = [200, 600, n_items]
    model = get_model(args, p_dims)
    optimizer = get_optimizer(model, args)
    criterion = get_criterion(args)

    # Training code
    best_n100 = -np.inf
    update_count = 0
    if args.model == "multivae":
        is_VAE = True
    else:
        is_VAE = False


    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train(model, criterion, optimizer, is_VAE)
        val_loss, n100, r10, r20, r50 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n100, r10, r20, r50))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))


        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_n100 = n100


    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, n100, r10, r20, r50 = evaluate(model, criterion, test_data_tr, test_data_te, is_VAE)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | r20 {:4.2f} | '
            'r50 {:4.2f}'.format(test_loss, n100, r10, r20, r50))
    print('=' * 89)

