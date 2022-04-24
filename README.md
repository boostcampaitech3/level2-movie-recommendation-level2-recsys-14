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

  





## 📝 Reference



### Multi-VAE with Side Information

https://arxiv.org/pdf/1807.05730.pdf



### H+vamp

https://github.com/psywaves/EVCF



### EASE

https://www.kaggle.com/code/lucamoroni/ease-implementation



### RecVAE

 https://github.com/ilya-shenbin/RecVAE/blob/master/model.py
