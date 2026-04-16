import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, streaming=True)
    return vectorstore, llm

vectorstore, llm = load_rag_components()

# 4. 프롬프트 템플릿 설정 (경량화된 기록 인지)
qa_system_prompt = """당신은 마인크래프트(Minecraft) 게임의 최고 전문가이자 친절한 지능형 가이드 챗봇입니다.
아래 제공된 [Context] 정보와 [대화 기록]을 참고하여 사용자의 질문에 답변하십시오.

답변을 작성할 때 다음 [가이드라인]을 반드시 준수하십시오:

[가이드라인]
1. 가독성: 마크다운(Markdown) 문법을 적극 활용하여 제목, 글머리 기호(-), 굵은 글씨(**) 등으로 깔끔하게 정리해서 답변하세요.
2. 조합법(제작) 질문: [Context]에 [1번칸:아이템명] 같은 3x3 제작 배치도 정보가 있다면, 
   반드시 '제작대 3x3 칸 기준'으로 위치를 명확히 풀어서 설명해주세요.
3. 몹(Mob) 질문: 체력, 공격력, 드롭 아이템, 스폰 장소, 특징 등 중요한 스펙을 요약해서 알려주세요.
4. 생물 군계/구조물 질문: 해당 지역의 특징, 발견할 수 있는 블록이나 전리품(상자), 출현하는 몹 위주로 설명해주세요.
5. 패치/업데이트 질문: 버전 역사 정보가 포함되어 있다면, 어느 버전에서 변경되었는지 명시해주세요.
6. ⚠️ 커뮤니티 팁/글리치 (면책 조항): [Context]에 없는 내용(글리치, 꼼수, 스피드런 팁 등)을 질문받으면, 당신의 사전 지식을 활용하여 답변하되, 반드시 답변 서두에 **[⚠️ 주의: 이 내용은 공식 위키에 없는 커뮤니티 팁/버그이며, 게임 버전에 따라 막혔거나 다를 수 있습니다]** 라고 명시하십시오. 공식 위키(Context)에 있는 내용은 경고문 없이 답변하세요.

[Context]
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

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

    # LangChain용 대화 기록 변환 및 이전 질문 추출
    chat_history = []
    prev_user_query = ""
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
            prev_user_query = msg["content"] # 가장 마지막 이전 질문 저장
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # RAG 파이프라인 가동 및 응답 스트리밍 출력
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("위키 DB를 검색 중입니다..."):
            # 💡 유저의 날카로운 지적 반영: 이전 질문을 무조건 합치면 주제 전환 시 검색이 꼬임!
            # 따라서 검색(Retrieval)은 오직 '현재 질문'으로만 수행하여 주제 전환에 완벽히 대응합니다.
            # 대명사 생략 질문("체력은?")은 LLM이 '이전 대화 기록(chat_history)'을 읽고 문맥을 파악하여 답변합니다.
            retrieved_docs = vectorstore.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "관련 정보를 찾을 수 없습니다."
            
            # 체인 조립 및 스트리밍 실행 (API 호출 1회로 감소 유지)
            chain = qa_prompt | llm
            for chunk in chain.stream({
                "context": context_text,
                "chat_history": chat_history,
                "input": user_query
            }):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        # 최종 응답 출력 및 기록 저장 (커서 제거)
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
