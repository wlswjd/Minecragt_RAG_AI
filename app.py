import os
import streamlit as st
from dotenv import load_dotenv
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 환경 변수 로드
load_dotenv()

# 게임별 설정 정보
GAME_CONFIG = {
    "마인크래프트": {
        "icon": "마크로고.webp",
        "collection": None,  # 기본 컬렉션 사용
        "description": "마인크래프트 공식 위키와 커뮤니티의 꿀팁들을 모두 모아, 게임 플레이 중 궁금한 점을 빠르고 정확하게 알려드리는 지능형 RAG 챗봇입니다.",
        "placeholder": "질문을 입력하세요 (예: 구리 곡괭이는 어떻게 만들어?)",
        "sources": "- 마인크래프트 공식 위키<br>- 나무위키 (팁/글리치)"
    },
    "발헤임": {
        "icon": None,
        "collection": "valheim",
        "description": "발헤임 나무위키 데이터를 학습한 RAG 챗봇입니다. 보스 공략, 장비 제작, 생물 군계 등 무엇이든 물어보세요.",
        "placeholder": "질문을 입력하세요 (예: 엘더는 어떻게 잡아?)",
        "sources": "- 나무위키 발헤임 문서"
    }
}

# 웹 UI 기본 설정 및 커스텀 CSS (모던 디자인)
st.set_page_config(page_title="Game RAG Guide", page_icon="마크로고.webp")

st.markdown("""
<style>
/* 사이드바 하단 고정 텍스트 */
[data-testid="stSidebar"] {
    position: relative;
}
.sidebar-footer {
    position: absolute;
    bottom: 20px;
    left: 20px;
    font-size: 0.8em;
    color: #a0a0a0;
}
</style>
""", unsafe_allow_html=True)

# 사이드바 - 게임 선택 드롭다운 및 설정
with st.sidebar:
    st.markdown("### 게임 선택")
    selected_game = st.selectbox(
        "게임을 선택하세요",
        list(GAME_CONFIG.keys()),
        label_visibility="collapsed"
    )
    
    # 게임 변경 감지 시 대화 초기화
    if "selected_game" not in st.session_state:
        st.session_state.selected_game = selected_game
    if st.session_state.selected_game != selected_game:
        st.session_state.selected_game = selected_game
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### ⚙️ 챗봇 설정")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<br>" * 15, unsafe_allow_html=True)
    st.markdown("---")
    
    # 게임별 데이터 소스 동적 출력
    config = GAME_CONFIG[selected_game]
    st.markdown(f"""
    <div class='sidebar-footer'>
    <b>현재 게임:</b><br>
    {selected_game}<br><br>
    <b>데이터 소스:</b><br>
    {config['sources']}<br><br>
    <b>AI 모델:</b><br>
    - Google Gemini 2.5 Flash
    </div>
    """, unsafe_allow_html=True)

# 선택된 게임에 따라 타이틀 및 설명 동적 변경
config = GAME_CONFIG[selected_game]
col1, col2 = st.columns([1, 8])
with col1:
    if config["icon"]:
        st.image(config["icon"], width=60)
with col2:
    st.title(f"{selected_game} 지능형 가이드")

st.markdown(f"**{config['description']}**")

# 모델 및 벡터 DB 로드 (캐싱 적용, 게임별 컬렉션 분리)
@st.cache_resource
def load_embeddings_and_llm():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, streaming=True)
    return embeddings, llm

@st.cache_resource
def load_vectorstore(_embeddings, collection_name):
    if collection_name:
        return Chroma(
            persist_directory="./chroma_db",
            embedding_function=_embeddings,
            collection_name=collection_name
        )
    return Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)

embeddings, llm = load_embeddings_and_llm()
vectorstore = load_vectorstore(embeddings, config["collection"])

# 프롬프트 템플릿 설정 (게임 동적 적용)
qa_system_prompt = f"""당신은 {selected_game} 게임의 최고 전문가이자 친절한 지능형 가이드 챗봇입니다.
아래 제공된 [Context] 정보와 [대화 기록]을 참고하여 사용자의 질문에 답변하십시오.

답변을 작성할 때 다음 [가이드라인]을 반드시 준수하십시오:

[가이드라인]
1. 가독성: 마크다운(Markdown) 문법을 적극 활용하여 제목, 글머리 기호(-), 굵은 글씨(**) 등으로 깔끔하게 정리해서 답변하세요.
2. 조합법(제작) 질문: 제작 방법이 있다면 필요한 재료와 위치를 명확히 풀어서 설명해주세요.
3. 몹/생물 질문: 체력, 공격력, 드롭 아이템, 스폰 장소, 특징 등 중요한 스펙을 요약해서 알려주세요.
4. 생물 군계/구조물 질문: 해당 지역의 특징, 발견할 수 있는 자원, 출현하는 몹 위주로 설명해주세요.
5. 패치/업데이트 질문: 버전 역사 정보가 포함되어 있다면, 어느 버전에서 변경되었는지 명시해주세요.
6. 커뮤니티 팁/글리치 (면책 조항): [Context]에 없는 내용(글리치, 꼼수, 스피드런 팁 등)을 질문받으면, 당신의 사전 지식을 활용하여 답변하되, 반드시 답변 서두에 **[⚠️ 주의: 이 내용은 공식 위키에 없는 커뮤니티 팁/버그이며, 게임 버전에 따라 막혔거나 다를 수 있습니다]** 라고 명시하십시오.

[Context]
{{context}}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 세션 상태 기반 대화 기록 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 기록 화면 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 및 챗봇 응답 처리
if user_query := st.chat_input(config["placeholder"]):
    # 사용자 질문 화면 표시 및 기록 저장
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # LangChain용 대화 기록 변환 및 이전 질문 추출
    chat_history = []
    prev_user_query = ""
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
            prev_user_query = msg["content"] # 마지막 이전 질문 저장
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # RAG 파이프라인 가동 및 응답 스트리밍 출력
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("위키 DB 검색 중..."):
            # 검색(Retrieval)은 현재 질문으로만 수행하여 주제 전환에 대응함.
            # 대명사 생략 질문은 LLM이 대화 기록을 읽고 문맥을 파악하여 답변함.
            retrieved_docs = vectorstore.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "관련 정보를 찾을 수 없습니다."
            
            # 체인 조립 및 스트리밍 실행 (API 호출 1회)
            chain = qa_prompt | llm
            for chunk in chain.stream({
                "context": context_text,
                "chat_history": chat_history,
                "input": user_query
            }):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        # 최종 응답 출력 및 기록 저장
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
