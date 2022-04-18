# Movie Recommendation Baseline Code

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec

## Installation

```
pip install -r requirements.txt
```

## How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py

   ```
1. Data looad
'get_user_seqs' user별 순차적으로 나열
test_rating_matrix는 유저별 마지막 일부 아이템을 빼서 구성

2. Dataset Load
- sampler가 dataset별로 다르다
- sampler: 배치를 끄집어낼 때 random은 섞어서(배치 뽑을 때 user들의 순서를 섞는것이다) sequential은 순차적(순서대로 채점해야하니까)으로 샘플링
- user seq : 각 유저가 어떤 영화를 봤는지
- negative sample 구성: item을 랜덤하게 이력이 없는 데이터로 뽑아줌
- padding: left padding으로 왼쪽에 0을 채워준다. max_seq보다 길이가 넘어가면 왼쪽의 데이터를 지워준다. 

3. Train
- sequence output(batch,seq_len,hiddensize): 예측한 값으로 hidden size는 하나의 영화에 대한 임베딩 사이즈이다

4. Self supervised learning
그냥 학습시키면 영화의 순서만 고려한게 아닌가 -> pretraining
다양한 관점에서 영화를 이해하도록 self-supervised learning을 수행
- 영화 장르, 순서, 영화순서와 장르, 영화 시리즈를 반영할 수 있도록 함, 해당 임베딩을 기반으로 더 좋은 sas rec모델을 학습시킴.
- Mutual information maximization에선 상호의존정보를 최대화한다. x를 아는 것이 y에 대한 불확실성을 얼마나 줄이는지, 그리고 이를 최대화시키도록 한다.  
- lower bound를 올리면 infoNCE가 cross entropy와 비슷하다. 따라서 CE loss로 훈련한다. 

5. tip
s3, als를 해보니 0.0844, 0.0877 -> als랑 성능차이가 크진 않다
-> static예측에 최적화된 모델은 다음 영화를 잘예측할 수 있을 것이다. 
-> sequential이랑 static앙상블? -> voting? 반반 나눠? 