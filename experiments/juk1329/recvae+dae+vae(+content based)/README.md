## 실험 배경
- sun1187님의 후시 분석 EDA에서 나온 결론 중에서 시청 이력이 적은 유저들을 모델이 잘 맞추지 못하는 경향이 있었습니다.
- 이러한 cold start 문제를 Content-based Filtering의 결과도 기존 모델들과 ensemble하여 효과를 보고자 했습니다.

---

## 실험 내용
- tf-idf를 사용하여 user_feature와 item_feature matrix를 구해 Content-based Filtering을 수행했습니다.
- 그리고 그 결과를 기존 모델들의 결과에 더해주는 과정을 수행했습니다.

---

## 실행 순서
0. item,user_feature_matrix.ipynb 파일 실행
1. ./VAE_DAE/train_dae.py 실행
2. ./VAE_DAE/train_vae.py 실행
3. train.py
4. inference.py

---

## 실험 결과

- Public Leaderboard와 Private Leaderboard 모두 낮은 성능을 보였습니다.
- Content-based Filtering을 활용하는 방법(user-item matrix의 변화, ensemble 적용 방법)에 대해 더 많은 고민이 필요해 보였던 실험이었습니다.
