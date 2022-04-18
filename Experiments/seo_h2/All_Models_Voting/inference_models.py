import argparse
import datetime
import json

import bottleneck as bn
from tqdm import tqdm

from sklearn.preprocessing import minmax_scale
from dataset import *
from models import *
from utils.utils import numerize_for_infer, denumerize_for_infer, naive_sparse2tensor

## 각종 파라미터 세팅
parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

parser.add_argument('--data', type=str, default='./data', help='Movielens data location')
parser.add_argument('--output_dir', type=str, default='./outputs', help='Logs directory')
parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/', help='Base Directory')
parser.add_argument('--split_mode', type=int, default=1, help='split_mode')
parser.add_argument('--fold_size', type=int, default=3000, help='fold_size')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--save_dir', type=str, default='./latest', help='directory to save the latest best model')
args = parser.parse_args()

## 배치사이즈 포함

### 데이터 준비
print("Inference Start!!")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_file_name = str(datetime.datetime.now())[0:10] + '_output_kfold.csv'
output_file = os.path.join(args.output_dir, output_file_name)

# Load Data
raw_data = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'), header=0)

ease_data = raw_data.loc[:, ["user", "item"]]

with open(os.path.join(args.data, 'show2id.json'), 'r', encoding="utf-8") as f:
    show2id = json.load(f)

with open(os.path.join(args.data, 'profile2id.json'), 'r', encoding="utf-8") as f:
    profile2id = json.load(f)
re_p2id = dict((int(v), int(k)) for k, v in profile2id.items())
re_s2id = dict((int(v), int(k)) for k, v in show2id.items())

infer_df = numerize_for_infer(raw_data, profile2id, show2id)
ease_df = numerize_for_infer(ease_data, profile2id, show2id)
ease_df['rating'] = [0.9] * len(ease_df)

pivot_table = ease_df.pivot_table(index=["uid"], columns=["sid"], values="rating")
X = pivot_table.to_numpy()
X = np.nan_to_num(X)

loader = DataLoader(args.data, args.split_mode, args.fold_size)
n_items = loader.load_n_items()

n_users = infer_df['uid'].max() + 1

rows, cols = infer_df['uid'], infer_df['sid']
data = sparse.csr_matrix((np.ones_like(rows),
                          (rows, cols)), dtype='float64',
                         shape=(n_users, n_items))

N = data.shape[0]
idxlist = list(range(N))

device = torch.device("cuda")


with open(os.path.join(args.save_dir, f'best_recvae{args.split_mode}_' + args.save), 'rb') as f:
    model_recvae = torch.load(f).to(device)

with open(os.path.join(args.save_dir, f'best_hvamp{args.split_mode}_' + args.save), 'rb') as f:
    model_hvamp = torch.load(f).to(device)

model_recvae.eval()
model_hvamp.eval()

ease_pred_list = None
rec_pred_list = None
hvamp_pred_list = None
ensem_pred_list = None

ease_model = EASE(300)
ease_model.train(X)

result = ease_model.forward(X[:, :])

with torch.no_grad():
    for start_idx in tqdm(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data_batch = data[idxlist[start_idx:end_idx]]
        data_tensor = naive_sparse2tensor(data_batch).to(device)

        recon_batch_recvae = model_recvae(data_tensor, calculate_loss=False).cpu().numpy()
        recon_batch_hvamp = model_hvamp.reconstruct_x(data_tensor).cpu().numpy()

        ensem_recon_batch = recon_batch_recvae * 1.3 + recon_batch_hvamp * 1.2
        ensem_recon_batch = minmax_scale(ensem_recon_batch.T).T
        ease_batch = result[[idxlist[start_idx:end_idx]]]
        ease_batch = minmax_scale(ease_batch.T).T

        ensem_recon_batch = ensem_recon_batch + ease_batch * 1.5

        ensem_recon_batch[data_batch.nonzero()] = -np.inf

        # Recall
        ensem_idx = bn.argpartition(-ensem_recon_batch, 10, axis=1)[:, :10]
        rec_idx = bn.argpartition(-recon_batch_recvae, 10, axis=1)[:, :10]
        ease_idx = bn.argpartition(-ease_batch, 10, axis=1)[:, :10]
        hvamp_idx = bn.argpartition(-recon_batch_hvamp, 10, axis=1)[:, :10]

        if start_idx == 0:
            ensem_pred_list = ensem_idx
            rec_pred_list = rec_idx
            hvamp_pred_list = hvamp_idx
            ease_pred_list = ease_idx

        else:
            ensem_pred_list = np.append(ensem_pred_list, ensem_idx, axis=0)
            rec_pred_list = np.append(rec_pred_list, rec_idx, axis=0)
            ease_pred_list = np.append(ease_pred_list, ease_idx, axis=0)
            hvamp_pred_list = np.append(hvamp_pred_list, hvamp_idx, axis=0)

# ensemble ---
user2 = []
item2 = []
for i_idx, arr_10 in enumerate(ensem_pred_list):
    user2.extend([i_idx] * 10)
    item2.extend(arr_10)

u2 = pd.DataFrame(user2, columns=['user'])
i2 = pd.DataFrame(item2, columns=['item'])
all2 = pd.concat([u2, i2], axis=1)

ans2 = denumerize_for_infer(all2, re_p2id, re_s2id)
ans2.columns = ['user', 'item']
new_ans2 = ans2.sort_values('user')

new_ans2.to_csv(output_file, index=False)
print('*****ensemble output saved!*****')

# recvae ---
user2 = []
item2 = []
for i_idx, arr_10 in enumerate(rec_pred_list):
    user2.extend([i_idx] * 10)
    item2.extend(arr_10)

u2 = pd.DataFrame(user2, columns=['user'])
i2 = pd.DataFrame(item2, columns=['item'])
all2 = pd.concat([u2, i2], axis=1)

ans2 = denumerize_for_infer(all2, re_p2id, re_s2id)
ans2.columns = ['user', 'item']
new_ans2 = ans2.sort_values('user')

output_file = os.path.join(args.output_dir, 'recvae.csv')
new_ans2.to_csv(output_file, index=False)
print('*****recvae output saved!*****')

# hvamp ---
user2 = []
item2 = []
for i_idx, arr_10 in enumerate(hvamp_pred_list):
    user2.extend([i_idx] * 10)
    item2.extend(arr_10)

u2 = pd.DataFrame(user2, columns=['user'])
i2 = pd.DataFrame(item2, columns=['item'])
all2 = pd.concat([u2, i2], axis=1)

ans2 = denumerize_for_infer(all2, re_p2id, re_s2id)
ans2.columns = ['user', 'item']
new_ans2 = ans2.sort_values('user')

output_file = os.path.join(args.output_dir, 'hvamp.csv')
new_ans2.to_csv(output_file, index=False)
print('*****hvamp output saved!*****')

# ease ---
user2 = []
item2 = []
for i_idx, arr_10 in enumerate(ease_pred_list):
    user2.extend([i_idx] * 10)
    item2.extend(arr_10)

u2 = pd.DataFrame(user2, columns=['user'])
i2 = pd.DataFrame(item2, columns=['item'])
all2 = pd.concat([u2, i2], axis=1)

ans2 = denumerize_for_infer(all2, re_p2id, re_s2id)
ans2.columns = ['user', 'item']
new_ans2 = ans2.sort_values('user')

output_file = os.path.join(args.output_dir, 'ease.csv')
new_ans2.to_csv(output_file, index=False)
print('*****ease output saved!*****')