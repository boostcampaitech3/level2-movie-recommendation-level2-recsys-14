from xml.etree.ElementTree import QName
import pandas as pd
import pickle
from scipy import sparse
import numpy as np
from utils.utils import numerize_for_infer, denumerize_for_infer

raw_data=pd.read_csv('train_ratings.csv')
ease= pd.read_csv('./outputs/ease_voting2.csv')
recvae= pd.read_csv('./outputs/recvae_voting.csv')
hvamp= pd.read_csv('./outputs/hvamp_voting.csv')
ensemble= pd.read_csv('./outputs/2022-04-14_output.csv')

with open('show2id.pkl','rb') as f:
    show2id= pickle.load(f)
with open('profile2id.pkl','rb') as f:
    profile2id= pickle.load(f)
re_p2id = dict((v, k) for k, v in profile2id.items())
re_s2id = dict((v, k) for k, v in show2id.items())

ensemble0= numerize_for_infer(ensemble, profile2id, show2id)
hvamp0 = numerize_for_infer(hvamp, profile2id, show2id)
recvae0=  numerize_for_infer(recvae, profile2id, show2id)
ease0=  numerize_for_infer(ease, profile2id, show2id)

n_users = raw_data['user'].nunique()
n_items = raw_data.item.nunique()
rows, cols = ensemble0['uid'],  ensemble0['sid']
ensemble_mat = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))

rows, cols = hvamp0['uid'], hvamp0['sid']
hvamp_mat = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))

rows, cols =recvae0['uid'], recvae0['sid']
recvae_mat = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))

rows, cols =ease0['uid'], ease0['sid']
ease_mat = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))

for u in ensemble0.uid.unique():
    items= list(ensemble0[ensemble0.uid ==u].sid)
    ensemble_mat[u,items]= [2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2]
for u in ease0.uid.unique():
    items= list(ease0[ease0.uid ==u].sid)
    ease_mat[u,items]= [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1]

total= ensemble_mat + hvamp_mat + recvae_mat + ease_mat
idx= np.argsort(-total.toarray())[:,:10]

user = []
item = []
for i_idx, arr_10 in enumerate(idx):
    user.extend([i_idx]*10)
    item.extend(arr_10)

u = pd.DataFrame(user, columns=['user'])
i = pd.DataFrame(item, columns=['item'])
all = pd.concat([u, i], axis=1)

ans = denumerize_for_infer(all, re_p2id, re_s2id)
ans.columns = ['user', 'item']
new_ans = ans.sort_values('user')
new_ans.reset_index(drop=True, inplace=True)

new_ans.to_csv("./outputs/final_voting5.csv", index=False)