# [Recsys-14] Movie Recommendation



## 📚 프로젝트 개요

### 📋 프로젝트 주제

Movielens-20M 기반의 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 선호할 만한 영화를 예측하는 최적의 모델을 개발합니다.





### 📈 프로젝트 목표

주어진 여러 추천 시스템 모델을 서로 비교해보고, 이를 통해 Sequential data와 static data 각각에 관한 추천 시스템의 차이를 이해합니다. 또한 대회에서 주어지는 데이터 분석을 통해 최적의 모델과 추천 방식을 찾고, 이를 바탕으로 모델의 성능과 normalized recall@10을 높이는 것을 목표로 합니다.





### 🗞 사용 데이터

**train_ratings.csv**

|  USER_ID  |  ITEM_ID  |         TIMESTAMP         |
| :-------: | :-------: | :-----------------------: |
| 사용자 ID | 아이템 ID | 사용자가 아이템을 본 시간 |



- 5,154,471개의 행으로 구성되었습니다.
- Implicit feedback 기반의 sequential recommendation 시나리오를 상정합니다.
- 사용자의 time-ordered sequence에서 일부 item이 누락(dropout)된 상태입니다.





### 🔑 프로젝트 특징

- timestamp를 바탕으로 사용자의 순차적인 이력과 Implicit feedback을 고려합니다.
- 1~5점 평점(Explicit feedback) 기반의 행렬을 사용한 협업 필터링 문제와 차별화되었습니다.





### 🛎 요구사항

사용자의 영화 시청 이력과 영화의 장르, 감독, 개봉연도 등 side-information을 활용하여 다음에 시청할 가능성이 높은 영화 상위 10개를 추천합니다.





### 📌 평가 방법

사전에 Movielens-20M 데이터에서 추출해놓은 ground-truth 아이템들을 고려하여 정규화된 Recall@10을 계산합니다.





## 프로젝트 구조

### Flow Chart

![mermaid-diagram-20220417234600](https://cdn.jsdelivr.net/gh/Glanceyes/Image-Repository/2022/04/24/20220424_1650794858.png)





### 디렉토리 구조

```html
/input/data
├─ eval                                          # 📁 평가 데이터
│   └─ sample_submission.csv                     # 📃 output sample
└─ train                                         # 📁 학습 데이터
    ├─ train_ratings.csv                         # 📄 주 학습 데이터
    ├─ Ml_item2attributes.json                   # 📃 item과 genre의 mapping 데이터
    ├─ titles.tsv                                # 🎬 영화 제목
    ├─ years.tsv                                 # 📆 영화 개봉년도
    ├─ directors.tsv                             # 📢 영화별 감독
    ├─ genres.tsv                                # 🎞 영화 장르
    └─ writers.tsv                               # 🖋 영화 작가

/code
├── README.md
├── data
│   ├── item_genre_emb.csv
│   ├── profile2id.json
│   ├── show2id.json
│   ├── test_te.csv
│   ├── test_tr.csv
│   ├── train.csv
│   ├── unique_sid.txt
│   ├── unique_uid.txt
│   ├── validation_te.csv
│   └── validation_tr.csv
├── dataset.py
├── inference.py
├── models.py
├── outputs
├── preprocessing.py
├── train_dae.py
├── train_hvamp.py
├── train_recvae.py
├── train_vae.py
└── utils
    ├── __init__.py
    ├── distributions.py
    ├── metrics.py
    ├── nn.py
    ├── optimizer.py
    └── utils.py
```





## 모델과 실험 내용

- 이번 프로젝트에서 실험한 단일 모델 종류는 Autoencoder 기반의 Multi-VAE, Multi-DAE, RecVAE, H+vamp, EASE를 사용했으며, 단일 모델 간 성능을 고려하여 최종 결과에 사용한 모델은 RecVAE, H+vamp, EASE입니다.
  - SASRec, BERT4Rec, S3Rec은 sequential data에 강한 모델로 알려져 있지만, 이번 프로젝트의 데이터는 단순 sequential data가 아니어서 예상보다 좋지 않은 결과가 나왔으며, 이렇나 이유로 좀 더 general recommendation에 적합한 모델을 탐색했다.
  - Multi-VAE, Multi-DAE에서 영화의 genre를 side-information으로 사용했습니다.
  - H+vamp는 기존의 VAE에 hierarchical stochastic unit과 vamp prior를 더한 모델이며, Multi-VAE의 한계를 극복하고자 적용한 모델입니다. 기존의 VAE에서 사전확률 분포인 $p(z)$를 추정하기 위해 사용한 정규 분포는 제약 사항이 많은데, 이를 ELBO를 최대화 하는 optimal prior의 근사치로 변경하여 좀 더 유연한 VampPrior를 사용했습니다. 결과적으로 Multi-VAE와 Multi-DAE를 앙상블 했을 때보다 성능이 향상되었다.
  - EASE(Embarrassingly Shallow Autoencoders for Sparse Data)는 파라미터의 해를 구하는 것이 closed form solution이며, 유저의 수가 증가할수록 loss function을 최소화하는 파라미터를 구하는 과정에서 쓰이는 Gram matrix의 계수가 커져서 파라미터를 구하는 error가 줄어들어 cold start problem에 강한 모델이다.
    - 특히 이번 프로젝트에서 다른 모델에 비해 단일 성능이 가장 좋게 나왔던 것은 아무래도 대회에서 사용한 데이터가 단순 sequential data가 아니라 일부분이 임의로 마스킹 되도록 전처리 되어서 좀 더 sparse data에 가까워져 cold start problem이 발생했는데, EASE는 이러한 현상에 강한 이유로 추측됩니다.
- Ensemble 시 RecVAE, H+vamp, EASE와 세 모델을 합친 Ensemble SOTA 모델로 Hard Voting을 진행했습니다.
  - 성능이 가장 좋았던 앙상블 모델 결과에 가장 많은 가중치를 부여했습니다.
  - 각 아이템 별 score가 voting에 반영될 수 있도록 순위에 따른 가중치를 부여했습니다.
  - 결과적으로 앙상블 모델의 비중을 2로 두고 순위를 반영한 것과, EASE에 순위를 반영한 것, Recvae와 h+vamp에는 아무 처리를 하지 않고 Hard Voting한 성능이 가장 좋았습니다.





## 💻 활용 도구 및 환경

- 코드 공유
    - GitHub
- 개발 환경
    - JupyterLab, VS Code, PyCharm
- 모델 성능 분석
    - Wandb
- 피드백 및 의견 공유
    - 카카오톡, Notion, Zoom  





## 👩🏻‍💻👨🏻‍💻 팀원 소개

<table>
   <tr>
      <td align="center">진완혁</td>
      <td align="center">박정규</td>
      <td align="center">김은선</td>
      <td align="center">이선호</td>
      <td align="center">이서희</td>
   </tr>
   <tr height="160px">
       <td align="center">
         <a href="https://github.com/wh4044">
            <!--<img height="120px" weight="120px" src="/pictures/jwh.png"/>-->
            <img width="100" alt="스크린샷 2022-04-19 오후 5 44 23" src="https://user-images.githubusercontent.com/70509258/163962658-548a3022-bcd3-40c7-8ca1-88c7c417e1d9.png">
         </a>
      <td align="center">
         <a href="https://github.com/juk1329">
            <!--<img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/80198264?s=400&v=4"/>-->
            <img width="85" alt="스크린샷 2022-04-19 오후 5 47 38" src="https://user-images.githubusercontent.com/70509258/163963317-f074768e-8976-42c5-a595-3c5be8310f48.png">
         </a>
      </td>
      </td>
      <td align="center">
         <a href="https://github.com/sun1187">
            <!--<img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/70509258?v=4"/>-->
        <img width="104" alt="스크린샷 2022-04-19 오후 5 48 35" src="https://user-images.githubusercontent.com/70509258/163963764-b66c30fc-de18-46ff-a432-3cec6cd5f9a8.png">
 </a>
      </td>
      <td align="center">
         <a href="https://github.com/Glanceyes">
          <!--<img height="120px" weight="120px" src="https://cdn.jsdelivr.net/gh/Glanceyes/Image-Repository/2022/03/24/20220324_1648093619.jpeg"/>-->
            <img width="96" alt="스크린샷 2022-04-19 오후 5 49 21" src="https://user-images.githubusercontent.com/70509258/163964338-4fc7e32a-d00e-46f5-a514-7d07ff14bcbc.png">
 </a>
      </td>
      <td align="center">
         <a href="https://github.com/seo-h2">
            <!--<img height="120px" weight="120px" src="/pictures/seoh2.png"/>-->
            <img width="102" alt="스크린샷 2022-04-19 오후 5 49 30" src="https://user-images.githubusercontent.com/70509258/163964515-eb89af1f-d9af-4c67-8ea9-b283383d3199.png">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">Truth</td>
      <td align="center">Juke</td>
      <td align="center">Sunny</td>
      <td align="center">Glaneyes</td>
      <td align="center">Brill</td>
   </tr>
</table>

  

| 팀원   | 역할                                                         |
| ------ | ------------------------------------------------------------ |
| 김은선 | EDA, Multi-vae/dae 모델 실험, 후시 분석을 통한 가설 검증과 Rule-based model 실험 |
| 박정규 | 모델 탐색, 성능이 좋은 모델에 EDA를 통한 여러 가설(year emb 추가, CB, 모델 별 추천 결과의 유사도 반영)을 실험 |
| 이서희 | EDA(유저 및 영화메타정보 분석), Rule-based model 및 voting 실험, 감독 및 장르선호도와 같은 side info 적용 실험 |
| 이선호 | Autoencoder 기반 모델 탐색(H+vamp, EASE) 및 분석, 코드 모듈화 및 리팩토링, k-fold 실험 |
| 진완혁 | 데이터 EDA, 알고리즘 탐색 및 적용, 하이퍼 파라미터 튜닝을 통한 모델 성능 개선, 모델간 성능 실험 |





## 📝 Reference



### Multi-VAE with Side Information

https://arxiv.org/pdf/1807.05730.pdf



### H+vamp

https://github.com/psywaves/EVCF



### EASE

https://www.kaggle.com/code/lucamoroni/ease-implementation



### RecVAE

 https://github.com/ilya-shenbin/RecVAE/blob/master/model.py
