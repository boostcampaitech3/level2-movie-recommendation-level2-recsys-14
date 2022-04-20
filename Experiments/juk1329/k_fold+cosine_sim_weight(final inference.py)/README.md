## 실험 배경
seo_h2님과 sun1187님의 후시 분석과 관련된 EDA에서 각 모델이 user에게 추천한 결과 중 상이한 경우도 많고, 적절하지 못한 상태로 ensemble 되는 것을 발견했습니다.

그래서 ensemble을 적용할 때의 적절한 가중치를 찾는 것이 중요해보여 이러한 실험을 진행하게 되었습니다.

---

## 실험 내용

- 기존에는 여러 모델에 ensemble을 적용할 때 가중치를 곱해서 더해주는 형식의 ensemble 방식을 이용했습니다.
- 이번 실험에서는 item,user_feature_matrix.ipynb에서 생성하는 item_feature와 user_feature를 이용하여 각 모델들을 ensemble 하기 전, 각 모델의 추천 결과와 추천한 그 유저의 user_feature 간 유사도를 가중치에 반영하였습니다.
- 유사도는 cosine similarity을 사용했습니다.
- 유사도를 가중치에 반영하는 과정
    - 기본적인 가중치를 설정해두고, 각 유저에 따라서 각 모델 간 유사도를 가중치에 더해주는 방식으로 각 유저마다 모델을 ensemble하는 가중치를 달리하는 실험이었습니다.

---

## 실행 순서

0. item,user_feature_matrix.ipynb 실행하여 item_feature.npy, user_feature.npy 생성
1. preprocessing.py --split_mode=3
2. train_recvae.py --split_mode=3
3. train_hvamp.py --split_mode=3
4. K_fold+cosine_sim_weight.py --split_mode=3

---

## 그 외

- 실험 결과 split_mode=3가 가장 높은 성능을 보였습니다.
- vae와 dae는 ensemble 과정에서 제외하는 것이 가장 높은 성능을 보였습니다.

---

## 실험 결과

- 대회 진행 중 Public Leaderboard에서는 가장 팀 내에서 높은 성능을 보여주진 못했지만, 대회 종료 후 Private Leaderboard에서는 팀 내에서 높은 성능을 보여주었습니다.