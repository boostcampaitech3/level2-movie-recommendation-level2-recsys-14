# Level2-recsys-14

- 프로젝트 주제
    - 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 모델 개발.

- 프로젝트 개요(프로젝트 구현 내용, 컨셉, 교육 내용과의 관련성 등)
    - 주요 데이터 - train_ratings.csv(USER_ID, ITEM_ID, TIMESTAMP)
        - 5,154,471개의 행으로 구성됨.
        - Implicit feedback 기반의 sequential recommendation 시나리오를 상정.
        - 사용자의 time-ordered sequence에서 일부 item이 누락(dropout)된 상황.
    - 컨셉
        - timestamp를 고려한 사용자의 순차적인 이력을 고려하고Implicit feedback을 고려.
        - 1-5점 평점(Explicit feedback) 기반의 행렬을 사용한 협업 필터링 문제와 차별화.
    - 구현 내용
        - 사용자의 영화 시청 이력, 장르 등 side-information을 활용하여 다음에 시청할 가능성이 높은 영화 상위 10개를 추천.
    
- 활용 장비 및 재료(개발 환경, 협업 tool 등)
    - 코드 공유
        - GitHub
    - 개발 환경
        - JupyterLab, VS Code, PyCharm
    - 모델 성능 분석
        - Wandb
    - 피드백 및 의견 공유
        - 카카오톡, Notion, Zoom
