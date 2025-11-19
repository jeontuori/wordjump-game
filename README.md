# 뜻말잇기 (Korean Word Jump Game)

FastText 한국어 임베딩을 활용해 **출발 단어에서 도착 단어까지 의미적으로 가까운 단어를 통해 이동하는 게임**입니다.  
단어 간 유사도는 FastText 벡터 기반 코사인 유사도를 사용합니다.

## 플레이 방법

1. 유사도가 0.12에서 0.13 사이인 출발 단어와 도착 단어가 자동으로 주어집니다.
2. 출발 단어에서부터 유사한 단어를 추리해서 입력하며 점프합니다.  
3. 각 점프는 유사도 30% 이상이어야 하며,  
   최종 도착 단어로의 점프는 40% 이상이어야 합니다.  
4. 지나온 단어들의 도착 단어와의 유사도 변화는 그래프로 표시됩니다.  
5. 도착 단어에 도달하면 축하 메시지와 함께 전체 경로가 표시됩니다.

## 입력 규칙

- 2~5음절의 **명사**만 입력할 수 있습니다.
- "사람", "문제"처럼 너무 일반적인 단어,  
  "해", "꽃"처럼 너무 많은 단어와 유사한 단어 등은 제외되었습니다.

## 실행 방법

### 환경 준비
Python 3.9+ 권장

```bash
pip install -r requirements.txt
```

### 실행

```bash
streamlit run app.py
```

## 기술 스택

- FastText 한국어 벡터 (cc. BY-SA)
- Python, Streamlit, Gensim
- Altair(그래프 시각화)

## 폴더 구조

```
wordjump-game/
│── app.py
│── ko_trimmed.vec
│── words.txt
│── requirements.txt
│── README.md
```