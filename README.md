# [Recsys-14] Movie Recommendation

<br/>

## π νλ‘μ νΈ κ°μ

### π νλ‘μ νΈ μ£Όμ 

Movielens-20M κΈ°λ°μ μ¬μ©μμ μν μμ²­ μ΄λ ₯ λ°μ΄ν°λ₯Ό λ°νμΌλ‘ μ¬μ©μκ° λ€μμ μ νΈν  λ§ν μνλ₯Ό μμΈ‘νλ μ΅μ μ λͺ¨λΈμ κ°λ°ν©λλ€.

<br/>

<br/>

### π νλ‘μ νΈ λͺ©ν

μ£Όμ΄μ§ μ¬λ¬ μΆμ² μμ€ν λͺ¨λΈμ μλ‘ λΉκ΅ν΄λ³΄κ³ , μ΄λ₯Ό ν΅ν΄ Sequential dataμ static data κ°κ°μ κ΄ν μΆμ² μμ€νμ μ°¨μ΄λ₯Ό μ΄ν΄ν©λλ€. λν λνμμ μ£Όμ΄μ§λ λ°μ΄ν° λΆμμ ν΅ν΄ μ΅μ μ λͺ¨λΈκ³Ό μΆμ² λ°©μμ μ°Ύκ³ , μ΄λ₯Ό λ°νμΌλ‘ λͺ¨λΈμ μ±λ₯κ³Ό normalized recall@10μ λμ΄λ κ²μ λͺ©νλ‘ ν©λλ€.

<br/>

<br/>

### π μ¬μ© λ°μ΄ν°

**train_ratings.csv**

|  USER_ID  |  ITEM_ID  |         TIMESTAMP         |
| :-------: | :-------: | :-----------------------: |
| μ¬μ©μ ID | μμ΄ν ID | μ¬μ©μκ° μμ΄νμ λ³Έ μκ° |



- 5,154,471κ°μ νμΌλ‘ κ΅¬μ±λμμ΅λλ€.
- Implicit feedback κΈ°λ°μ sequential recommendation μλλ¦¬μ€λ₯Ό μμ ν©λλ€.
- μ¬μ©μμ time-ordered sequenceμμ μΌλΆ itemμ΄ λλ½(dropout)λ μνμλλ€.

<br/>

<br/>

### π νλ‘μ νΈ νΉμ§

- timestampλ₯Ό λ°νμΌλ‘ μ¬μ©μμ μμ°¨μ μΈ μ΄λ ₯κ³Ό Implicit feedbackμ κ³ λ €ν©λλ€.
- 1~5μ  νμ (Explicit feedback) κΈ°λ°μ νλ ¬μ μ¬μ©ν νμ νν°λ§ λ¬Έμ μ μ°¨λ³νλμμ΅λλ€.

<br/>

<br/>

### π μκ΅¬μ¬ν­

μ¬μ©μμ μν μμ²­ μ΄λ ₯κ³Ό μνμ μ₯λ₯΄, κ°λ, κ°λ΄μ°λ λ± side-informationμ νμ©νμ¬ λ€μμ μμ²­ν  κ°λ₯μ±μ΄ λμ μν μμ 10κ°λ₯Ό μΆμ²ν©λλ€.<br/>

<br/>

### π νκ° λ°©λ²

μ¬μ μ Movielens-20M λ°μ΄ν°μμ μΆμΆν΄λμ ground-truth μμ΄νλ€μ κ³ λ €νμ¬ μ κ·νλ Recall@10μ κ³μ°ν©λλ€.

<br/>

<br/>

## νλ‘μ νΈ κ΅¬μ‘°

### Flow Chart

![mermaid-diagram-20220417234600](https://cdn.jsdelivr.net/gh/Glanceyes/Image-Repository/2022/04/24/20220424_1650794858.png)

<br/>

<br/>

### λλ ν λ¦¬ κ΅¬μ‘°

```html
/input/data
ββΒ evalΒ Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β #Β πΒ νκ°Β λ°μ΄ν°
βΒ Β Β ββ sample_submission.csv                     # π output sample
ββΒ trainΒ Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β Β #Β πΒ νμ΅Β λ°μ΄ν°
 Β Β Β ββΒ train_ratings.csv                         # π μ£Ό νμ΅ λ°μ΄ν°
    ββΒ Ml_item2attributes.json                   # π itemκ³Ό genreμ mapping λ°μ΄ν°
    ββ titles.tsv                                # π¬ μν μ λͺ©
    ββ years.tsv                                 # π μν κ°λ΄λλ
    ββ directors.tsv                             # π’ μνλ³ κ°λ
    ββ genres.tsv                                # π μν μ₯λ₯΄
    ββ writers.tsv                               # πΒ μν μκ°

/code
βββ README.md
βββ data
βΒ Β  βββ item_genre_emb.csv
βΒ Β  βββ profile2id.json
βΒ Β  βββ show2id.json
βΒ Β  βββ test_te.csv
βΒ Β  βββ test_tr.csv
βΒ Β  βββ train.csv
βΒ Β  βββ unique_sid.txt
βΒ Β  βββ unique_uid.txt
βΒ Β  βββ validation_te.csv
βΒ Β  βββ validation_tr.csv
βββ dataset.py
βββ inference.py
βββ models.py
βββ outputs
βββ preprocessing.py
βββ train_dae.py
βββ train_hvamp.py
βββ train_recvae.py
βββ train_vae.py
βββ utils
    βββ __init__.py
    βββ distributions.py
    βββ metrics.py
    βββ nn.py
    βββ optimizer.py
    βββ utils.py
```

<br/>

<br/>

## λͺ¨λΈκ³Ό μ€ν λ΄μ©

- μ΄λ² νλ‘μ νΈμμ μ€νν λ¨μΌ λͺ¨λΈ μ’λ₯λ Autoencoder κΈ°λ°μ Multi-VAE, Multi-DAE, RecVAE, H+vamp, EASEλ₯Ό μ¬μ©νμΌλ©°, λ¨μΌ λͺ¨λΈ κ° μ±λ₯μ κ³ λ €νμ¬ μ΅μ’ κ²°κ³Όμ μ¬μ©ν λͺ¨λΈμ RecVAE, H+vamp, EASEμλλ€.
  - SASRec, BERT4Rec, S3Recμ sequential dataμ κ°ν λͺ¨λΈλ‘ μλ €μ Έ μμ§λ§, μ΄λ² νλ‘μ νΈμ λ°μ΄ν°λ λ¨μ sequential dataκ° μλμ΄μ μμλ³΄λ€ μ’μ§ μμ κ²°κ³Όκ° λμμ΅λλ€. κ·Έλμ μ’ λ general recommendationμ μ ν©ν λͺ¨λΈμ νμνμ΅λλ€.
  - Multi-VAE, Multi-DAEμμ μνμ genreλ₯Ό side-informationμΌλ‘ μ¬μ©νμ΅λλ€.
  - H+vampλ κΈ°μ‘΄μ VAEμ hierarchical stochastic unitκ³Ό vamp priorλ₯Ό λν λͺ¨λΈμ΄λ©°, Multi-VAEμ νκ³λ₯Ό κ·Ήλ³΅νκ³ μ μ μ©ν λͺ¨λΈμλλ€. κΈ°μ‘΄μ VAEμμ μ¬μ νλ₯  λΆν¬μΈ $p(z)$λ₯Ό μΆμ νκΈ° μν΄ μ¬μ©ν μ κ· λΆν¬λ μ μ½ μ¬ν­μ΄ λ§μλ°, μ΄λ₯Ό ELBOλ₯Ό μ΅λν νλ optimal priorμ κ·Όμ¬μΉλ‘ λ³κ²½νμ¬ μ’ λ μ μ°ν VampPriorλ₯Ό μ¬μ©νμ΅λλ€. κ²°κ³Όμ μΌλ‘ Multi-VAEμ Multi-DAEλ₯Ό μμλΈ νμ λλ³΄λ€ μ±λ₯μ΄ ν₯μλμλ€.
  - EASE(Embarrassingly Shallow Autoencoders for Sparse Data)λ νλΌλ―Έν°μ ν΄λ₯Ό κ΅¬νλ κ²μ΄ closed form solutionμ΄λ©°, μ μ μ μκ° μ¦κ°ν μλ‘ loss functionμ μ΅μννλ νλΌλ―Έν°λ₯Ό κ΅¬νλ κ³Όμ μμ μ°μ΄λ Gram matrixμ κ³μκ° μ»€μ Έμ νλΌλ―Έν°λ₯Ό κ΅¬νλ errorκ° μ€μ΄λ€μ΄ cold start problemμ κ°ν λͺ¨λΈλ‘ μλ €μ Έ μμ΅λλ€.
    - νΉν μ΄λ² νλ‘μ νΈμμ λ€λ₯Έ λͺ¨λΈμ λΉν΄ λ¨μΌ μ±λ₯μ΄ κ°μ₯ μ’κ² λμλ κ²μ μλ¬΄λλ λνμμ μ¬μ©ν λ°μ΄ν°κ° λ¨μ sequential dataκ° μλλΌ μΌλΆλΆμ΄ μμλ‘ λ§μ€νΉ λλλ‘ μ μ²λ¦¬ λμ΄μ μ’ λ sparse dataμ κ°κΉμμ Έ cold start problemμ΄ λ°μνλλ°, EASEλ μ΄λ¬ν νμμ κ°ν μ΄μ λ‘ μΆμΈ‘λ©λλ€.
- Ensemble μ RecVAE, H+vamp, EASEμ μΈ λͺ¨λΈμ ν©μΉ Ensemble SOTA λͺ¨λΈλ‘ Hard Votingμ μ§ννμ΅λλ€.
  - μ±λ₯μ΄ κ°μ₯ μ’μλ μμλΈ λͺ¨λΈ κ²°κ³Όμ κ°μ₯ λ§μ κ°μ€μΉλ₯Ό λΆμ¬νμ΅λλ€.
  - κ° μμ΄ν λ³ scoreκ° votingμ λ°μλ  μ μλλ‘ μμμ λ°λ₯Έ κ°μ€μΉλ₯Ό λΆμ¬νμ΅λλ€.
  - κ²°κ³Όμ μΌλ‘ μμλΈ λͺ¨λΈμ λΉμ€μ 2λ‘ λκ³  μμλ₯Ό λ°μν κ²κ³Ό, EASEμ μμλ₯Ό λ°μν κ², Recvaeμ h+vampμλ μλ¬΄ μ²λ¦¬λ₯Ό νμ§ μκ³  Hard Votingν μ±λ₯μ΄ κ°μ₯ μ’μμ΅λλ€.

<br/>

<br/>

## π» νμ© λκ΅¬ λ° νκ²½

- μ½λ κ³΅μ 
    - GitHub
- κ°λ° νκ²½
    - JupyterLab, VS Code, PyCharm
- λͺ¨λΈ μ±λ₯ λΆμ
    - Wandb
- νΌλλ°± λ° μκ²¬ κ³΅μ 
    - μΉ΄μΉ΄μ€ν‘, Notion, Zoom  

<br/>

<br/>

## π©π»βπ»π¨π»βπ» νμ μκ°

<table>
   <tr>
      <td align="center">μ§μν</td>
      <td align="center">λ°μ κ·</td>
      <td align="center">κΉμμ </td>
      <td align="center">μ΄μ νΈ</td>
      <td align="center">μ΄μν¬</td>
   </tr>
   <tr height="160px">
       <td align="center">
         <a href="https://github.com/wh4044">
            <!--<img height="120px" weight="120px" src="/pictures/jwh.png"/>-->
            <img width="100" alt="μ€ν¬λ¦°μ· 2022-04-19 μ€ν 5 44 23" src="https://user-images.githubusercontent.com/70509258/163962658-548a3022-bcd3-40c7-8ca1-88c7c417e1d9.png">
         </a>
      <td align="center">
         <a href="https://github.com/juk1329">
            <!--<img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/80198264?s=400&v=4"/>-->
            <img width="85" alt="μ€ν¬λ¦°μ· 2022-04-19 μ€ν 5 47 38" src="https://user-images.githubusercontent.com/70509258/163963317-f074768e-8976-42c5-a595-3c5be8310f48.png">
         </a>
      </td>
      </td>
      <td align="center">
         <a href="https://github.com/sun1187">
            <!--<img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/70509258?v=4"/>-->
        <img width="104" alt="μ€ν¬λ¦°μ· 2022-04-19 μ€ν 5 48 35" src="https://user-images.githubusercontent.com/70509258/163963764-b66c30fc-de18-46ff-a432-3cec6cd5f9a8.png">
 </a>
      </td>
      <td align="center">
         <a href="https://github.com/Glanceyes">
          <!--<img height="120px" weight="120px" src="https://cdn.jsdelivr.net/gh/Glanceyes/Image-Repository/2022/03/24/20220324_1648093619.jpeg"/>-->
            <img width="96" alt="μ€ν¬λ¦°μ· 2022-04-19 μ€ν 5 49 21" src="https://user-images.githubusercontent.com/70509258/163964338-4fc7e32a-d00e-46f5-a514-7d07ff14bcbc.png">
 </a>
      </td>
      <td align="center">
         <a href="https://github.com/seo-h2">
            <!--<img height="120px" weight="120px" src="/pictures/seoh2.png"/>-->
            <img width="102" alt="μ€ν¬λ¦°μ· 2022-04-19 μ€ν 5 49 30" src="https://user-images.githubusercontent.com/70509258/163964515-eb89af1f-d9af-4c67-8ea9-b283383d3199.png">
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

  <br/>

| νμ   | μ­ν                                                          |
| ------ | ------------------------------------------------------------ |
| κΉμμ  | EDA, Multi-vae/dae λͺ¨λΈ μ€ν, νμ λΆμμ ν΅ν κ°μ€ κ²μ¦κ³Ό Rule-based model μ€ν |
| λ°μ κ· | λͺ¨λΈ νμ, μ±λ₯μ΄ μ’μ λͺ¨λΈμ EDAλ₯Ό ν΅ν μ¬λ¬ κ°μ€(year emb μΆκ°, CB, λͺ¨λΈ λ³ μΆμ² κ²°κ³Όμ μ μ¬λ λ°μ)μ μ€ν |
| μ΄μν¬ | EDA(μ μ  λ° μνλ©νμ λ³΄ λΆμ), Rule-based model λ° voting μ€ν, κ°λ λ° μ₯λ₯΄μ νΈλμ κ°μ side info μ μ© μ€ν |
| μ΄μ νΈ | Autoencoder κΈ°λ° λͺ¨λΈ νμ(H+vamp, EASE) λ° λΆμ, μ½λ λͺ¨λν λ° λ¦¬ν©ν λ§, k-fold μ€ν |
| μ§μν | λ°μ΄ν° EDA, μκ³ λ¦¬μ¦ νμ λ° μ μ©, νμ΄νΌ νλΌλ―Έν° νλμ ν΅ν λͺ¨λΈ μ±λ₯ κ°μ , λͺ¨λΈκ° μ±λ₯ μ€ν |

<br/>

<br/>

## π Reference

<br/>

### Multi-VAE with Side Information

https://arxiv.org/pdf/1807.05730.pdf

<br/>

### H+vamp

https://github.com/psywaves/EVCF

<br/>

### EASE

https://www.kaggle.com/code/lucamoroni/ease-implementation

<br/>

### RecVAE

 https://github.com/ilya-shenbin/RecVAE
