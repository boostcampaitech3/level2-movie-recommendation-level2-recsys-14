import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from gensim.models import Word2Vec
import torch
from sklearn.metrics import pairwise_distances
from collections import Counter

# 데이터셋 ---
recvae7=pd.read_csv('recvae_dir_genre7.csv') #10개 추천
recvae7_50=pd.read_csv('recvae_dir_genre7_50.csv') #50개 추천
low_tr= pd.read_csv('/opt/ml/input/data/train/train_ratings.csv') # 학습에 사용된 전체 데이터셋
lowest_result_users= list(pd.read_csv('recvae_lowest_user.csv')['user']) # 못맞춘 유저군
#선호도 데이터 ---
dir_pref= np.load('pref_item0.npy') #numerize, 유저별 특정 감독의 영화에 대한 선호도
dir_pref[dir_pref==2] = 10
genre_pref= np.load("lst.npy")
# 워드임베딩 모델 ---
word_model = Word2Vec.load('item2vec_word2vecSg_20180328')
word_vectors = word_model.wv
# 학습 모델 ---
recvae= torch.load('recvae7.pt') # 모델로드


# 못맞춘 유저군 추천 데이터 추출 ---
lowest_reco_result0= recvae7_50[recvae7_50.user.isin(lowest_result_users)]

# recvae에서 유사도 추출 ---
emb= recvae.decoder.weight
emb= emb.detach().cpu().numpy()
sim= 1-pairwise_distances(emb, metric="cosine")
sim[sim == 1]= -np.inf # 자기자신 제외

with open('profile2id_check.pkl', 'rb') as f:
    profile2id = pickle.load(f)

with open('show2id_check.pkl', 'rb') as f:
    show2id = pickle.load(f)

re_p2id = dict((v, k) for k, v in profile2id.items())
re_s2id = dict((v, k) for k, v in show2id.items())

# 장르 및 감독 선호도 반영했을 때 실행 ---
def reflect_pref(dic, dic2, user):
    key= dic.keys()
    for k in key:
        dic[k] *= dir_pref[profile2id[user]][show2id[k]] # 감독선호
        dic[k] += genre_pref[profile2id[user]][show2id[k]] # 장르선호

    key2= dic2.keys()
    for k in key2:
        dic2[k] *= dir_pref[profile2id[user]][show2id[k]] # 감독선호
        dic2[k] += genre_pref[profile2id[user]][show2id[k]] # 장르선호
    
    return dic,dic2

# 유저가 시청한 영화와 유사한 아이템들로 상위 10개 재추천 ---
def similarities(user, reflect=True):
    low_tr_item= low_tr[low_tr.user == user]['item']
    rec_item= lowest_reco_result0[lowest_reco_result0.user==user]['item']

    low_tr_item2= low_tr[low_tr.user == user]['item'].apply(lambda x: show2id[x]) #recvae sim적용위한 numerize
    rec_item2= lowest_reco_result0[lowest_reco_result0.user==user]['item'].apply(lambda x: show2id[x])

    leng= len(low_tr_item)
    rec_l= len(rec_item)
    if leng > rec_l:
        sim_df= pd.DataFrame(data={'low_tr':list(low_tr_item.sort_values()), 'rec_item': list(rec_item.sort_values()) + [7000] * abs(leng-rec_l)})
    elif leng < rec_l:
        sim_df= pd.DataFrame(data={'low_tr':list(low_tr_item.sort_values())+ [7000] * abs(leng-rec_l), 'rec_item': list(rec_item.sort_values())})
    else:
        sim_df= pd.DataFrame(data={'low_tr':list(low_tr_item.sort_values()), 'rec_item': list(rec_item.sort_values())})

    leng2= len(low_tr_item2)
    rec_l2= len(rec_item2)
    if leng2 > rec_l2:
        sim_df2= pd.DataFrame(data={'low_tr':list(low_tr_item2.sort_values()), 'rec_item': list(rec_item2.sort_values()) + [7000] * abs(leng2-rec_l2)})
    elif leng2 < rec_l2:
        sim_df2= pd.DataFrame(data={'low_tr':list(low_tr_item2.sort_values())+ [7000] * abs(leng2-rec_l2), 'rec_item': list(rec_item2.sort_values())})
    else:
        sim_df2= pd.DataFrame(data={'low_tr':list(low_tr_item2.sort_values()), 'rec_item': list(rec_item2.sort_values())})

    sim_df['sim']=0
    sim_df2['sim']=0

    sim_pivot= sim_df.pivot_table(index='low_tr', columns='rec_item', values='sim')
    sim_pivot2= sim_df2.pivot_table(index='low_tr', columns='rec_item', values='sim')
    if sum([7000 in sim_pivot.index]) >=1:
        sim_pivot.drop(7000, axis=0, inplace=True)
    elif sum([7000 in sim_pivot.columns]) >=1:
        sim_pivot.drop(7000, axis=1, inplace=True)

    if sum([7000 in sim_pivot2.index]) >=1:
        sim_pivot2.drop(7000, axis=0, inplace=True)
    elif sum([7000 in sim_pivot2.columns]) >=1:
        sim_pivot2.drop(7000, axis=1, inplace=True)

    dic={}
    idx=list(sim_pivot.index)
    for col in sim_pivot.columns:
        for idx in sim_pivot.index:
            try: 
                sim_pivot[col][idx]= word_vectors.similarity(str(col),str(idx))
            except:
                sim_pivot[col][idx]= 0

        dic[col]= sim_pivot[col].sum()

    dic2={}
    idx2=list(sim_pivot2.index)
    for col in sim_pivot2.columns:
        sim_pivot2[col]=sim[col][idx2]
        dic2[re_s2id[col]]= sim_pivot2[col].sum() # denum
    

    if reflect==True:
        dic, dic2= reflect_pref(dic,dic2,user)
    dic3= Counter(dic2) + Counter(dic)
    temp= sorted(dic3.items(), key = lambda item: item[1], reverse=True)[:10] # 원래 아이디
    items= list(map(lambda x:x[0], temp))

    return user, items


user2 = []
item2 = []

for u in tqdm(lowest_result_users):
    user, items = similarities(u, reflect=True)
    user2.extend([user]*10)
    item2.extend(items)

u2 = pd.DataFrame(user2, columns=['user'])
i2 = pd.DataFrame(item2, columns=['item'])
all2 = pd.concat([u2, i2], axis=1)

sub= recvae7[recvae7.user.isin(lowest_result_users)].index
recvae7.drop(sub, axis=0, inplace=True)
final= pd.concat([recvae7, all2], axis=0)

final.to_csv('recvae_low_process2.csv', index=False)
print("새로운 추천 파일 생성 완료!!")