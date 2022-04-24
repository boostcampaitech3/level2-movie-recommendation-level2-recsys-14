# [Glanceyes] Movie Recommendation 프로젝트 개인 회고

<br/>

## 개인 학습 목표

Autoencoder 기반의 여러 모델을 학습하고 각 모델의 특징을 분석하면서 모델별로 어떠한 요인으로 인해 서로 다른 추천 결과를 출력하는지에 관해 집중적으로 탐구합니다. 또한 팀원과의 협업을 통해 모델의 코드를 모듈화하여 배포하는 과정에 익숙해지는 것을 목표로 합니다.

<br/>

<br/>

## 학습 목표를 위한 노력과 시도

### Autoencoder 기반의 SOTA 모델 학습하기

베이스라인 코드로 제공되는 Multi-DAE와 Multi-VAE 뿐만이 아니라 [paperswithcode.com](http://paperswithcode.com/)의 Movielens 20M 데이터에서 좋은 성능을 보인 H+vamp와 EASE 모델을 대회에 적용하기 위해 논문을 공부했습니다. 기존의 autoencoder 기반의 모델이 갖고 있던 사전 확률 분포의 제약에 관한 개선, 딥 러닝을 위해 많은 파라미터를 학습하지 않고 손실함수를 최소화하는 한 개의 파라미터를 closed form으로 찾는 방법 등 모델을 개선하고자 하는 시도를 학습하면서 베이스라인 모델에서 발전시킬 수 있을 만한 방향을 모색했습니다.

<br/>

### Recbole 라이브러리로 모델 간 성능 비교

앞에서 사용한 모델 뿐만이 아니라 추가로 NeuMF, Item KNN 등 다른 general recommendation 모델을 사용하기 위해 Recbole 라이브러리를 사용했습니다. 옵션과 하이퍼파라미터 설정만 변경하여 빠르게 모델 학습과 검증을 실행할 수 있어서 막바지에 모델 성능을 신속하게 비교할 수 있었습니다.

<br/>

<br/>

## 한계와 개선방향

후반으로 갈수록 모델 예측 결과에 집착하면서 모델 자체를 제대로 분석하지 못했는데, 단일 모델 성능이 가장 좋았던 EASE 모델 분석에 집중하고 정작 RecVAE와 H+vamp 분석에는 시간을 쏟지 못한 점이 아쉬웠습니다.

또한 Recbole 라이브러리를 대회 진행 후반에 알게 되어서 Recbole 실행에 충분한 시간을 쓰지 못하여 autoencoder 기반 모델 외의 다른 SOTA 모델을 실험해 보지 못했는데, 다음에는 Recbole 등 외부 라이브러리를 초반에 적극적으로 활용하여 모델 탐색의 시간을 줄이고자 합니다.

Title Embedding을 사용한 성능 향상 효과가 유의미하지 않아서 결국 팀원과 회의하여 없애는 방식으로 결정했는데, 제목보다는 영화의 감독과 개봉연도를 side information으로 사용하는 방법을 더 탐구해봐야 하지 않았나 하는 생각이 듭니다.