import numpy as np
import pandas as pd
import pickle
import bottleneck as bn

with open('profile2id.pkl', 'rb') as f:
    profile2id = pickle.load(f)

with open('show2id.pkl', 'rb') as f:
    show2id = pickle.load(f)

def denumerize_for_infer(tp, re_p2id, re_s2id):
    uid2 = tp['user'].apply(lambda x: re_p2id[x])
    sid2 = tp['item'].apply(lambda x: re_s2id[x])
    return pd.DataFrame(data={'uid': uid2, 'sid': sid2}, columns=['uid', 'sid'])


def infer(model, num_k, X):
    # 비교 데이터 적용 + 결과 저장
    result = num_k.forward(X[:, :])
    result[X.nonzero()] = -np.inf
    pred_list = bn.argpartition(-result, num_k, axis=1)[:, :num_k]

    user2 = []
    item2 = []
    for i_idx, arr_10 in enumerate(pred_list):
        user2.extend([i_idx]*num_k)
        item2.extend(arr_10)

    u2 = pd.DataFrame(user2, columns=['user'])
    i2 = pd.DataFrame(item2, columns=['item'])
    all2 = pd.concat([u2, i2], axis=1)

    re_p2id = dict((int(v), int(k)) for k, v in profile2id.items())
    re_s2id = dict((int(v), int(k)) for k, v in show2id.items())

    ans2 = denumerize_for_infer(all2, re_p2id, re_s2id)
    ans2.columns = ['user', 'item']
    new_ans2 = ans2.sort_values('user')

    new_ans2.to_csv('EASE_check_output.csv', index=False)
    
def save_ans(check_x):
    # answer 저장
    df = pd.DataFrame(check_x)

    cols = df.columns.values
    mask = df.gt(0.0).values
    out = [cols[x].tolist() for x in mask]

    user = []
    item = []

    for i in range(len(out)):
        user.extend([i]*len(out[i]))
        item.extend(out[i])

    u = pd.DataFrame(user, columns=['user'])
    i = pd.DataFrame(item, columns=['item'])
    all = pd.concat([u, i], axis=1)

    re_p2id = dict((v, k) for k, v in profile2id.items())
    re_s2id = dict((v, k) for k, v in show2id.items())

    def de_numerize(tp, re_p2id, re_s2id):
        uid = tp['user'].apply(lambda x: re_p2id[x])
        sid = tp['item'].apply(lambda x: re_s2id[x])
        return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

    real_ans = de_numerize(all, re_p2id, re_s2id)
    real_ans.columns = ['user', 'item']
    real_new_ans = real_ans.sort_values('user')
    real_new_ans.to_csv('EASE_check_answer.csv', index=False)
    