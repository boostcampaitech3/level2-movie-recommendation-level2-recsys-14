import argparse
import datetime
import json
import warnings

import bottleneck as bn
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from dataset import *
from models import *
from utils.utils import numerize_for_infer, denumerize_for_infer, naive_sparse2tensor

warnings.filterwarnings(action='ignore')


# Set Arguments
parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

parser.add_argument('--data', type=str, default='./data', help='Movielens data location')
parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/', help='Base Directory')
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
parser.add_argument('--output_dir', type=str, default='./outputs', help='directory to save an inference output')
parser.add_argument('--ease_lambda', type=int, default=300, help='hyperparameter lambda for EASE')
args = parser.parse_args()

device = torch.device("cuda")


# Import Data
print("Inference Start!!")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_file_name = str(datetime.datetime.now())[0:10] + '_output.csv'
output_file = os.path.join(args.output_dir, output_file_name)


# Load Data
raw_data = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'), header=0)

with open(os.path.join(args.data, 'show2id.json'), 'r', encoding="utf-8") as f:
    show2id = json.load(f)

with open(os.path.join(args.data, 'profile2id.json'), 'r', encoding="utf-8") as f:
    profile2id = json.load(f)

infer_df = numerize_for_infer(raw_data, profile2id, show2id)

loader = DataLoader(args.data)
n_items = loader.load_n_items()
n_users = infer_df['uid'].max() + 1

rows, cols = infer_df['uid'], infer_df['sid']
data = sparse.csr_matrix((np.ones_like(rows),(rows, cols)), dtype='float64', shape=(n_users, n_items))

num_data = data.shape[0]
index_list = list(range(num_data))


# Load Model
with open(os.path.join(args.save_dir, 'best_dae_' + args.save), 'rb') as f:
    model_dae = torch.load(f).to(device)

with open(os.path.join(args.save_dir, 'best_vae_' + args.save), 'rb') as f:
    model_vae = torch.load(f).to(device)

with open(os.path.join(args.save_dir, 'best_recvae_' + args.save), 'rb') as f:
    model_recvae = torch.load(f).to(device)

with open(os.path.join(args.save_dir, 'best_hvamp_' + args.save), 'rb') as f:
    model_hvamp = torch.load(f).to(device)

model_recvae.eval()
model_dae.eval()
model_vae.eval()
model_hvamp.eval()


# Train EASE
ease_df = infer_df
ease_df['rating'] = [1] * len(ease_df)

pivot_table = ease_df.pivot_table(index=["uid"], columns=["sid"], values="rating")
ease_input = pivot_table.to_numpy()
ease_input = np.nan_to_num(ease_input)


ease_model = EASE(args.ease_lambda)
ease_model.train(ease_input)
ease_result = ease_model.forward(ease_input[:, :])


# Infer
pred_list = None

with torch.no_grad():
    for start_index in tqdm(range(0, num_data, args.batch_size)):
        end_index = min(start_index + args.batch_size, num_data)
        data_batch = data[index_list[start_index:end_index]]
        data_tensor = naive_sparse2tensor(data_batch).to(device)

        recon_batch = model_recvae(data_tensor, calculate_loss=False)
        recon_batch_dae = model_dae(data_tensor)
        recon_batch_vae, _, _ = model_vae(data_tensor)
        recon_batch_hvamp = model_hvamp.reconstruct_x(data_tensor)

        recon_batch = recon_batch.cpu().numpy()
        recon_batch_dae = recon_batch_dae.cpu().numpy()
        recon_batch_vae = recon_batch_vae.cpu().numpy()
        recon_batch_hvamp = recon_batch_hvamp.cpu().numpy()

        recon_batch = recon_batch + recon_batch_dae + recon_batch_vae + recon_batch_hvamp
        recon_batch = minmax_scale(recon_batch.T).T

        ease_batch = ease_result[[index_list[start_index:end_index]]]
        ease_batch = minmax_scale(ease_batch.T).T

        recon_batch = recon_batch + ease_batch * 1.5

        recon_batch[data_batch.nonzero()] = -np.inf

        idx = bn.argpartition(-recon_batch, 10, axis=1)[:, :10]
        if start_index == 0:
            pred_list = idx
        else:
            pred_list = np.append(pred_list, idx, axis=0)


user_infer = []
item_infer = []

for user_index, pred_item in enumerate(pred_list):
    user_infer.extend([user_index] * 10)
    item_infer.extend(pred_item)

u_infer = pd.DataFrame(user_infer, columns=['user'])
i_infer = pd.DataFrame(item_infer, columns=['item'])
all_infer = pd.concat([u_infer, i_infer], axis=1)

id2profile = dict((int(v), int(k)) for k, v in profile2id.items())
id2show = dict((int(v), int(k)) for k, v in show2id.items())

answer = denumerize_for_infer(all_infer, id2profile, id2show)
answer.columns = ['user', 'item']
sorted_answer = answer.sort_values('user')

sorted_answer.to_csv(output_file, index=False)
print('Output Saved!!')
