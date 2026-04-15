import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 1. 환경 변수 로드
load_dotenv()

# 2. 웹 UI 기본 설정
st.set_page_config(page_title="Minecraft RAG Guide", page_icon="⛏️")
st.title("⛏️ 마인크래프트 지능형 가이드")
st.markdown("로컬 마인크래프트 위키 데이터를 활용한 RAG 챗봇입니다.")

# 3. 모델 및 벡터 DB 로드 (캐싱을 통해 최초 1회만 로드)
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return vectorstore, llm

vectorstore, llm = load_rag_components()

# 4. 프롬프트 템플릿 설정
prompt_template = """
당신은 마인크래프트 게임의 지능형 가이드 챗봇입니다.
아래 제공된 [Context]만을 사용하여 사용자의 [Question]에 답변하십시오.
Context에 없는 내용은 절대 추측해서 답변하지 마십시오.

[Context]
{context}

[Question]
{question}
"""
prompt = PromptTemplate.from_template(prompt_template)

# 5. 세션 상태(Session State)를 이용한 대화 기록 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 기록을 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 사용자 입력 및 챗봇 응답 처리
if user_query := st.chat_input("질문을 입력하세요 (예: 구리 곡괭이는 어떻게 만들어?)"):
    # 사용자의 질문을 화면에 표시하고 기록에 저장
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # RAG 파이프라인 가동 및 응답 출력
    with st.chat_message("assistant"):
        with st.spinner("위키 DB를 검색 중입니다..."):
            # 벡터 DB 검색
            retrieved_docs = vectorstore.similarity_search(user_query, k=1)
            context_text = retrieved_docs[0].page_content if retrieved_docs else "관련 정보를 찾을 수 없습니다."
            
            # 프롬프트 조립 및 LLM 요청
            final_prompt = prompt.format(context=context_text, question=user_query)
            response = llm.invoke(final_prompt)
            answer = response.content
            
            # 결과 화면 출력 및 기록에 저장
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})