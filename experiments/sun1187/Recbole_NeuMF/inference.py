import torch
import numpy as np
import pandas as pd
from recbole.quick_start import load_data_and_model
from tqdm import tqdm
import bottleneck as bn

def de_numerize(tp, re_p2id, re_s2id):
    uid = tp['user'].apply(lambda x: re_p2id[x])
    sid = tp['item'].apply(lambda x: re_s2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':

    model_path='saved/NeuMF-Apr-14-2022_04-45-38.pth'
    print(model_path)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
        
    device = config.final_config_dict['device']
    
    uid2user = dataset.field2id_token['user'] 
    iid2item = dataset.field2id_token['item']
    
    matrix = dataset.inter_matrix(form='csr')
    
    #inferene용 데이터 준비
    unseen = np.where(matrix.toarray()[:,:]==0)
    unseen_u = unseen[0]
    unseen_i = unseen[1]
    
    x_u_torch = torch.IntTensor(unseen_u)
    x_i_torch = torch.IntTensor(unseen_i)

    x_u_torch = x_u_torch.to(device)
    x_i_torch = x_i_torch.to(device)

    # batchsize만큼 split하기
    batchsize = 800
    u_splits = x_u_torch.split(batchsize, dim=0)
    i_splits = x_i_torch.split(batchsize, dim=0)

    # user id, predict item id 저장 변수 
    pred_list = []
    user_list = []
    item_list = []

    model.eval()
    for i in tqdm(range(len(u_splits))):
        batch_idx = i

        output2 = model(u_splits[batch_idx], i_splits[batch_idx])
        rating_pred = output2.cpu().data.numpy().copy() 

        batch_user_index = u_splits[batch_idx].cpu().numpy()
        batch_item_index = i_splits[batch_idx].cpu().numpy()

        # 예측값 저장
        pred_list.extend(rating_pred) 
        user_list.extend(batch_user_index)
        item_list.extend(batch_item_index)

    mat = np.zeros((31361, 6808))
    indices = user_list, item_list#[:100]
    values = pred_list#[:100]
    mat[indices] = values

    idx = bn.argpartition(-mat, 10, axis=1)[:, :10]

    user = []
    item = []
    for i_idx, arr_10 in enumerate(idx):
        user.extend([i_idx]*10)
        item.extend(arr_10)

    u = pd.DataFrame(user, columns=['user'])
    i = pd.DataFrame(item, columns=['item'])
    all = pd.concat([u, i], axis=1)

    ans = de_numerize(all, uid2user, iid2item)
    ans.columns = ['user', 'item']
    ans[10:].to_csv('submit.csv')

    print('inference done!')
