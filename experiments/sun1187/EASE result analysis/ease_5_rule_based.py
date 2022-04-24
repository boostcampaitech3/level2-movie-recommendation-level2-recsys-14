import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore') 

#바꿔야될 파일
EASE_5output = pd.read_csv('EASE_5output.csv')
EASE_10output = pd.read_csv('EASE_10output.csv')
EASE_30output = pd.read_csv('EASE_30output.csv')
#EASE_50output = pd.read_csv('EASE_50output.csv')
#바꾸지 않아도 될 파일: (경로 변경하기)
inter_user = pd.read_csv('hvamp_rec_ease.csv')
lowest_history_year = pd.read_csv('ease_lowest_history_year.csv')
lowest_users_genre_prefer = pd.read_csv('ease_lowest_users_genre_prefer.csv')
years = pd.read_csv("/opt/ml/input/data/train/years.tsv", delimiter='\t')
gen = pd.read_csv("/opt/ml/input/data/train/genres.tsv", delimiter='\t')


print('ease rule based 시작!!!')
#전처리
lowest_history_year.drop('Unnamed: 0', axis=1, inplace=True)
lowest_users_genre_prefer.set_index('users', inplace=True)

##rule base 적용
lowest_users = inter_user['0'].values #lowest_history_year.user.unique()
item = []
user = []
genre_interest = []

for i in tqdm(range(len(lowest_users))):
    user_idx = lowest_users[i]

    ## 유저가 많이 본 영화들의 개봉년도 적용
    std_re = round(np.std(lowest_history_year[lowest_history_year.user==user_idx].year.values), 3)
    mean_re = round(np.mean(lowest_history_year[lowest_history_year.user==user_idx].year.values), 3)
    range_b = mean_re-std_re
    range_a = mean_re+std_re

    candidates = pd.DataFrame(years[(range_b <= years.year) & (years.year <= range_a)].item.values, columns=['item'])
    candidates['user'] = user_idx

    ## 유저가 본 영화 제거
    history_items = lowest_history_year[lowest_history_year.user==user_idx].item.values # 본 영화 리스트
    candidates = candidates[~candidates.item.isin(history_items)]

    # reconbatch 결과 top30만 고르기
    candidates = candidates[candidates.item.isin(EASE_30output[EASE_30output.user == user_idx].item.values)]

    cnt = 2
    while len(candidates) < 10:
        if std_re == 0.0:
            std_re = 0.001
        range_b = mean_re-cnt*std_re
        range_a = mean_re+cnt*std_re

        candidates = pd.DataFrame(years[(range_b <= years.year) & (years.year <= range_a)].item.values, columns=['item'])
        candidates['user'] = user_idx

        ## 유저가 본 영화 제거
        history_items = lowest_history_year[lowest_history_year.user==user_idx].item.values # 본 영화 리스트
        candidates = candidates[~candidates.item.isin(history_items)]

        # reconbatch 결과 top30만 고르기
        candidates = candidates[candidates.item.isin(EASE_30output[EASE_30output.user == user_idx].item.values)]
        cnt += 1

    candidates['genre_interest'] = 0.0
    candidates.reset_index(inplace=True, drop=True)
    for i in range(len(candidates)):
        u = candidates['user'][i]
        its = candidates['item'][i]

        tmp_gen = gen[gen.item == its].genre.values

        interest = 0
        for k in tmp_gen:
            interest += lowest_users_genre_prefer.loc[u][k]

        candidates['genre_interest'][i] = interest/len(tmp_gen)

    new_candi = candidates.sort_values('genre_interest', ascending=False)[:10]

    item.extend(new_candi.item)
    user.extend(new_candi.user)
    genre_interest.extend(new_candi.genre_interest)

# 결과 데이터프레임으로 정리: lowest 유저군 rule 기반 top5
lowest_reco_new_result = pd.DataFrame(user, columns=['user'])
lowest_reco_new_result['item'] = item
# lowest 유저군에 속하지 않는 추천 결과들과 합치기
fin_result = EASE_10output[~EASE_10output.user.isin(lowest_users)] # lowest에 속하지 않는 유저들
fin_result = pd.concat([fin_result, lowest_reco_new_result], axis=0)
lowest_model_5_out = EASE_5output[EASE_5output.user.isin(lowest_users)] # lowest 유저군 모델기반 top5
fin_result = pd.concat([fin_result, lowest_model_5_out], axis=0)

fin_result = fin_result.sort_values('user', ascending=True)
fin_result.reset_index(inplace=True, drop=True)
fin_result.to_csv('ease_rule_based_module.csv')
print('ease rule based 완료!!!')
