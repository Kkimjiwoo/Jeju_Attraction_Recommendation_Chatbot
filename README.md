# Jeju_Attraction_Recommendation_Chatbot
### 주제 : LLM을 활용한 제주도 맛집 추천 챗봇
팀원 : 김지우, 성수린, 이예진

## Overview
### `Outline`
- 리뷰 데이터를 기반으로 동행자, 방문 목적, 주변 관광지를 고려한 사용자의 니즈에 최적화된 음식점을 추천하는 서비스를 제공
  
## Project Process
### `Data Collection`
- 신한카드의 제주 가맹점 이용 데이터
- 네이버 맵 가맹점 리뷰 데이터

### `Chatbot`
- LLM_카테고리 분류
  추천형 검색형 질문을 분류하기 위한 LLM 생성
- LLM_키워드 추출
  키워드 추출을 위한 LLM 생성
- LLM_최종 응답 작성
  최종 응답작성을 위한 LLM 생성

각 LLM에 맞는 페르소나와 프롬프트를 작성하여, 최종 응답을 생성함
