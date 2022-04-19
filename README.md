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

