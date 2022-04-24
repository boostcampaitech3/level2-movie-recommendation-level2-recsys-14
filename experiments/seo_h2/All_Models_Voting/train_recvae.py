import argparse
import datetime
import random
from copy import deepcopy

import torch.optim as optim
import wandb

from dataset import *
from models import RecVAE
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data', type=str)
parser.add_argument('--log_dir', type=str, default='./logs', help='Logs directory')
parser.add_argument('--split_mode', type=int, default=1, help='data split mode')
parser.add_argument('--fold_size', type=int, default=3000, help='fold size for k-fold')
parser.add_argument('--hidden_dim', type=int, default=600)
parser.add_argument('--latent_dim', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0007)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--n_enc_epochs', type=int, default=3)
parser.add_argument('--n_dec_epochs', type=int, default=1)
parser.add_argument('--not_alternating', type=bool, default=False)
parser.add_argument('--save', type=str, default='model.pt', help='file name to save the final model')
parser.add_argument('--save_dir', type=str, default='./latest', help='directory to save the latest best model')
args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0")


def evaluate(model, data_tr, data_te, metrics, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()

    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    for m in metrics:
        m['score'] = []

    with torch.no_grad():
        for start_idx in range(0, e_N, batch_size):
            end_idx = min(start_idx + batch_size, e_N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            ratings_pred = model(data_tensor, calculate_loss=False).cpu().detach().numpy()

            if not (data_tr is data_te):
                ratings_pred[data.nonzero()] = -np.inf

            for m in metrics:
                m['score'].append(m['metric'](ratings_pred, heldout_data, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()

    return [x['score'] for x in metrics]


def train(model, opts, train_data, batch_size, beta, gamma, dropout_rate):
    model.train()
    np.random.shuffle(idxlist)
    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(start_idx + batch_size, N)

        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)

        for optimizer in opts:
            optimizer.zero_grad()

        _, loss = model(data, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
        loss.backward()

        for optimizer in opts:
            optimizer.step()


print("Split mode {} (1 ~ 6)".format(args.split_mode))
# data = get_data(args.data, args.split_mode)
# train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data

loader = DataLoader(args.data, args.split_mode, args.fold_size)

n_items = loader.load_n_items()
train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')
test_data_tr, test_data_te = loader.load_data('test')

N = train_data.shape[0]  # 25360
idxlist = list(range(N))

model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': train_data.shape[1]
}

metrics = [
    {'metric': ndcg, 'k': 100},
    {'metric': recall, 'k': 10},
    {'metric': recall, 'k': 20},
    {'metric': recall, 'k': 50}
]

best_recall = -np.inf

model = RecVAE(**model_kwargs).to(device)
model_best = RecVAE(**model_kwargs).to(device)

learning_kwargs = {
    'model': model,
    'train_data': train_data,
    'batch_size': args.batch_size,
    'beta': args.beta,
    'gamma': args.gamma
}

decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)

log_dir_name = str(datetime.datetime.now())[0:10] + f'_recvae{args.split_mode}'
log_dir = os.path.join(args.log_dir, log_dir_name)
log_dir = increment_path(log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# -- wandb initialize with configuration
wandb.init(config={"model": 'RecVAE',
                   "batch_size": args.batch_size,
                   "lr": args.lr,
                   "epochs": args.n_epochs,
                   })

for epoch in range(args.n_epochs):

    if args.not_alternating:
        train(opts=[optimizer_encoder, optimizer_decoder], dropout_rate=0.5, **learning_kwargs)
    else:
        train(opts=[optimizer_encoder], dropout_rate=0.5, **learning_kwargs)
        model.update_prior()
        train(opts=[optimizer_decoder], dropout_rate=0, **learning_kwargs)

    train_scores = evaluate(model, train_data, train_data, metrics)
    valid_scores = evaluate(model, vad_data_tr, vad_data_te, metrics)

    print('-' * 70)
    print(f'epoch {epoch} | previous best recall: {best_recall}')

    print('training set')
    for metric, score in zip(metrics, train_scores):
        print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")

    valid_recall = 0
    log_dict = {}

    print('validation set')
    for metric, score in zip(metrics, valid_scores):
        if metric['metric'].__name__ == 'recall' and metric['k'] == 10:
            valid_recall = score
        print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")
        log_dict[f"recvae_{metric['metric'].__name__}{metric['k']}"] = score

    wandb.log(log_dict)

    if valid_recall > best_recall:
        best_recall = valid_recall
        model_best.load_state_dict(deepcopy(model.state_dict()))
        with open(os.path.join(log_dir, f'best_recvae{args.split_mode}_' + args.save), 'wb') as f:
            torch.save(model, f)
        print(f"best_model saved!! (save {best_recall})")
    print('-' * 70 + '\n')

with open(os.path.join(args.save_dir, f'best_recvae{args.split_mode}_' + args.save), 'wb') as f:
    torch.save(model_best, f)

final_scores = evaluate(model_best, test_data_tr, test_data_te, metrics)

for metric, score in zip(metrics, final_scores):
    print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")
