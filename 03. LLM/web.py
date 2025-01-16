
import streamlit as st
import pandas as pd
import torch
import os
import json
from PIL import Image
import subprocess
from langchain.vectorstores import Chroma
# Google Drive 마운트
# from google.colab import drive
# drive.mount('/content/drive')

# Google Drive에서 defs.py 파일 복사
subprocess.run(['cp', '/content/drive/MyDrive/빅콘테스트/05.코드모음/06.streamlit코드/ngrok_in_colab/defs.py', './'])

# defs.py에서 필요한 함수 가져오기
from defs import (clear_chat_history, category_classification, search_main, recommendation_main, other_main)

import google.generativeai as genai
import os

# 파일에서 API 키 읽어오기
with open('/content/drive/MyDrive/빅콘테스트/05.코드모음/06.streamlit코드/ngrok_in_colab/apikey.txt', 'r') as f:
    api_key = f.read().strip()  # 공백 제거

# 환경 변수에 API 키 설정
os.environ['GEMINI_API_KEY'] = api_key

##################################################################
#chatbot UI
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")
st.title('제주도 음식점 탐방!')
st.subheader("누구랑 제주도 왔나요? 맞춤 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")

with st.sidebar:
    st.title('<옵션을 선택하면 빠르게 추천해드려요!>')
    st.write("")

    st.subheader('지역을 선택하세요! 해당 지역의 맛집을 찾아드릴께요.')
    st.write("")

    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')  # 제주시 체크박스
    local_seogwipo_city = st.checkbox('서귀포시')  # 서귀포시 체크박스
    st.write("")

    # 둘 다 체크되면 False로 설정
    if local_jeju_city and local_seogwipo_city:
        local_jeju_city = False
        local_seogwipo_city = False

    # PNG 이미지 삽입 (제주도 지도.png 이미지 삽입!!!!!!!!!!!!)
    image = Image.open('/content/drive/MyDrive/빅콘테스트/05.코드모음/data/제주도 지도.png')  # 이미지 파일 경로
    st.image(image, caption='제주도 지도', use_container_width=True)  # 사이드바에 이미지 삽입

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if the initial assistant message has been displayed
if "message_displayed" not in st.session_state:
    st.session_state.message_displayed = False

# Display the initial assistant message only once
if not st.session_state.message_displayed:
    st.session_state.messages.append({"role": "assistant", "content": "여행 중 제주 맛집 추천이 필요하신가요? 저희 챗봇은 사용자의 필요에 맞춘 맛집 정보를 제공합니다."})
    st.session_state.message_displayed = True  # Mark message as displayed

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "여행 중 제주 맛집 추천이 필요하신가요? 저희 챗봇은 사용자의 필요에 맞춘 맛집 정보를 제공합니다."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
########################### 검색형 데이터 csv ###########################
# 해당 데이터 경로로 변경 하세요!!
path = r'/content/drive/MyDrive/빅콘테스트/05.코드모음/data/JEJU_MCT_DATA_v2(12월)_v2.csv'
raw = pd.read_csv(path, index_col = 0)
df = raw.copy()

#########################임베딩 모델 로드##############################
from langchain.embeddings import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
#############################ChromaDB##############################
# ChromaDB 불러오기
# 해당 데이터 경로로 변경 하세요!!
recommendation_store = Chroma(
    collection_name='jeju_store_mct_keyword_6',
    embedding_function=embedding_function,
    persist_directory= r'/content/drive/MyDrive/빅콘테스트/05.코드모음/data/DB1'
)
# metadata 설정
metadata = recommendation_store.get(include=['metadatas'])
###########################################사용자 입력 쿼리################################################
# 사용자 입력에 따른 검색
# 사용자 입력에 따른 검색
if user_input := st.chat_input('사용자 특성이나 여행 동반자, 위치와 같은 조건을 입력해보세요.'):
    # 사용자 입력 메시지 저장 및 출력
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 음식점 찾기 스피너
    with st.spinner("음식점을 찾는 중입니다..."):
        # 입력 분류
        classification = category_classification(user_input)

        # 검색형 분류
        if classification['Classification'] == '검색형':
            print('분류 >> 검색형')
            response = search_main(user_input, df)

        # 추천형 분류
        elif classification['Classification'] == '추천형':
            print('분류 >> 추천형')

            seogwipo_dong_list = ['서귀포시', '서귀포', '서귀동', '대포동', '하원동', '하예동', '대정읍', '상효동', '호근동', '도순동', '중문동', '동홍동', '남원읍', '성산읍', '하효동', '강정동', '색달동', '회수동', '보목동', '안덕면', '법환동', '월평동', '토평동', '신효동', '표선면', '상예동', '서호동', '서홍동']
            jeju_dong_list = ['제주시', '이호이동', '화북일동', '용담이동', '연동', '용담삼동', '도련이동', '애월', '애월읍', '도평동', '한경면', '오라삼동', '회천동', '아라이동', '오라일동', '화북이동', '건입동', '이도이동', '삼양삼동', '외도일동', '용담일동', '해안동', '이도일동', '오라이동', '삼도이동', '삼양일동', '도두이동', '노형동', '구좌읍', '오등동', '외도이동', '아라일동', '도련일동', '도두일동', '삼도일동', '우도면', '조천읍', '일도일동', '삼양이동', '봉개동', '한림읍', '월평동', '내도동', '영평동', '일도2동', '일도이동', '도남동', '이호일동']

            # 제주시와 서귀포시 선택 조건에 따른 안내 메시지 처리
            if local_jeju_city and not local_seogwipo_city and any(value in user_input for value in seogwipo_dong_list): #제주시 체크시
                response = "제주시에 있는 음식점만 추천해드릴 수 있어요. 서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요."
            elif local_seogwipo_city and not local_jeju_city and any(value in user_input for value in jeju_dong_list): #서귀포시 체크시
                response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요. 제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
            else:
                response = recommendation_main(user_input, local_jeju_city, local_seogwipo_city, recommendation_store)

        # 기타 분류
        else:
            print('분류 >> 기타')
            response = other_main(user_input)

        # 챗봇 응답 메시지 설정
        assistant_response = response

    # 챗봇 응답 메시지 저장 및 출력
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)
