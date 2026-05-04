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
        "collection": "langchain",
        "description": "마인크래프트 공식 위키와 커뮤니티의 꿀팁들을 모두 모아, 게임 플레이 중 궁금한 점을 빠르고 정확하게 알려드리는 지능형 RAG 챗봇입니다.",
        "placeholder": "질문을 입력하세요 (예: 구리 곡괭이는 어떻게 만들어?)",
        "sources": "- 마인크래프트 공식 위키<br>- 나무위키 (팁/글리치)",
        "accent": "#3CAF55"
    },
    "발헤임": {
        "icon": "발헤임로고.png",
        "collection": "valheim",
        "description": "발헤임 나무위키 + 커뮤니티 공략글까지 학습한 RAG 챗봇입니다. 보스 공략, 장비 제작, 빌드 노하우까지 무엇이든 물어보세요.",
        "placeholder": "질문을 입력하세요 (예: 검은숲 트롤은 어떻게 잡아?)",
        "sources": "- 나무위키 발헤임 문서<br>- 디시 발헤임 갤러리 (개념글 공략)",
        "accent": "#E07B3C"
    }
}

# 웹 UI 기본 설정
st.set_page_config(
    page_title="Game RAG Guide",
    page_icon="마크로고.webp",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기 설정 (게임 변경 감지용 사전 처리)
if "selected_game" not in st.session_state:
    st.session_state.selected_game = list(GAME_CONFIG.keys())[0]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바 - 게임 선택 및 설정
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>🎮 Game Wiki AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle'>나만의 게임 가이드 챗봇</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("##### 게임 선택")
    selected_game = st.selectbox(
        "게임을 선택하세요",
        list(GAME_CONFIG.keys()),
        label_visibility="collapsed",
        key="game_selector"
    )
    
    if st.session_state.selected_game != selected_game:
        st.session_state.selected_game = selected_game
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("##### 챗봇 제어")
    if st.button("🔄  대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    config = GAME_CONFIG[selected_game]
    st.markdown(f"""
    <div class='sidebar-info'>
        <div class='info-row'><span class='info-label'>현재 게임</span><span class='info-value'>{selected_game}</span></div>
        <div class='info-row'><span class='info-label'>AI 모델</span><span class='info-value'>Gemini 2.5 Flash</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='sidebar-footer'>
        <div class='footer-title'>데이터 소스</div>
        <div class='footer-content'>{config['sources']}</div>
        <div class='footer-divider'></div>
        <div class='footer-copy'>© 2026 Game RAG AI</div>
    </div>
    """, unsafe_allow_html=True)

# 동적 CSS (게임별 강조 색상)
config = GAME_CONFIG[selected_game]
st.markdown(f"""
<style>
/* 전체 레이아웃 여백 및 폰트 */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 900px;
}}

/* 헤더 카드 */
.header-card {{
    background: linear-gradient(135deg, rgba(30,37,50,0.85), rgba(43,49,62,0.7));
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid {config['accent']};
    display: flex;
    align-items: center;
    gap: 1.2rem;
}}
.header-card img {{
    height: 56px;
    border-radius: 8px;
}}
.header-card .title {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #fafafa;
    margin: 0;
}}
.header-card .subtitle {{
    font-size: 0.9rem;
    color: #c0c0c0;
    margin-top: 4px;
    line-height: 1.5;
}}

/* 사이드바 브랜딩 */
.sidebar-brand {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #fafafa;
    padding: 8px 0 4px 0;
}}
.sidebar-subtitle {{
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 8px;
}}

/* 사이드바 정보 카드 */
.sidebar-info {{
    background-color: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.85em;
    color: #d0d0d0;
}}
.info-row {{
    display: flex;
    justify-content: space-between;
    margin: 4px 0;
}}
.info-label {{
    color: #888;
}}
.info-value {{
    color: #fafafa;
    font-weight: 600;
}}

/* 사이드바 하단 푸터 */
[data-testid="stSidebar"] {{
    position: relative;
}}
.sidebar-footer {{
    margin-top: 30px;
    font-size: 0.78em;
    color: #888;
}}
.footer-title {{
    color: #aaa;
    font-weight: 600;
    margin-bottom: 6px;
    font-size: 0.85em;
}}
.footer-content {{
    color: #888;
    line-height: 1.6;
}}
.footer-divider {{
    border-top: 1px solid #333;
    margin: 14px 0;
}}
.footer-copy {{
    color: #666;
    font-size: 0.9em;
}}

/* 사이드바 버튼 호버 효과 */
[data-testid="stSidebar"] .stButton button {{
    background-color: rgba(255,255,255,0.05);
    color: #e0e0e0;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.2s ease;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background-color: {config['accent']}33;
    border-color: {config['accent']};
    color: #fafafa;
}}

/* 채팅 입력창 강조 */
[data-testid="stChatInput"] {{
    border-top: 1px solid {config['accent']}33;
}}
</style>
""", unsafe_allow_html=True)

# 헤더 영역 (로고 + 타이틀 + 설명을 카드로 통합)
icon_html = f"<img src='data:image/png;base64,{__import__('base64').b64encode(open(config['icon'],'rb').read()).decode()}' />" if config['icon'] and os.path.exists(config['icon']) else ""

st.markdown(f"""
<div class='header-card'>
    {icon_html}
    <div>
        <div class='title'>{selected_game} 지능형 가이드</div>
        <div class='subtitle'>{config['description']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

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

[Context의 출처 구분]
각 Context 조각은 "출처: ..." 라는 머리말로 시작하며, 다음과 같이 신뢰도와 성격이 다릅니다.
- **공식 위키 / 나무위키**: 검증된 게임 공식/정리 정보. 수치, 제작법, 기본 메커니즘 등 사실 기반 답변에 사용하세요.
- **커뮤니티 발헤임 갤러리 공략**: 실제 유저들의 플레이 경험에서 나온 노하우, 빌드, 보스 공략, 꿀팁입니다. 공식이 아니므로 패치 등에 따라 달라질 수 있습니다.

답변을 작성할 때 다음 [가이드라인]을 반드시 준수하십시오:

[가이드라인]
1. 가독성: 마크다운(Markdown) 문법을 적극 활용하여 제목, 글머리 기호(-), 굵은 글씨(**) 등으로 깔끔하게 정리해서 답변하세요.
2. 조합법(제작) 질문: 제작 방법이 있다면 필요한 재료와 위치를 명확히 풀어서 설명해주세요.
3. 몹/생물 질문: 체력, 공격력, 드롭 아이템, 스폰 장소, 특징 등 중요한 스펙을 요약해서 알려주세요.
4. 생물 군계/구조물/지역 질문: 해당 지역의 특징, 발견할 수 있는 자원, 출현하는 몹 위주로 설명해주세요.
5. 패치/업데이트 질문: 버전 역사 정보가 포함되어 있다면, 어느 버전에서 변경되었는지 명시해주세요.
6. 보스 공략 / 빌드 / 실전 팁 질문: 디시 갤러리 공략 Context가 있다면 적극 활용해 단계별 전략, 추천 장비, 주의할 패턴 등을 구체적으로 정리해주세요.
7. **커뮤니티 출처 활용 시 면책 조항 (매우 중요)**: 답변에 디시인사이드 갤러리 공략 Context를 사용한 경우, 답변 **맨 위**에 다음 문구를 반드시 포함하십시오.
   `> 💡 이 답변에는 디시인사이드 발헤임 갤러리 유저들의 실전 공략이 포함되어 있습니다. 공식 정보가 아니므로 게임 버전이나 상황에 따라 다를 수 있습니다.`
   그리고 답변 본문에서 해당 정보를 인용할 때는 자연스럽게 `(유저 공략글 "[글 제목]" 참고)` 와 같이 출처 글의 제목을 함께 표기해주세요.
8. **Context에 없는 내용**: [Context]에 관련 정보가 전혀 없는 글리치/꼼수 등을 질문받으면, 당신의 사전 지식을 활용해 답변하되 반드시 서두에 다음 문구를 명시하십시오.
   `> ⚠️ 주의: 이 내용은 제공된 자료에 없는 일반 지식 기반 답변이며, 게임 버전에 따라 막혔거나 다를 수 있습니다.`

[Context]
{{context}}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 이전 대화 기록 화면 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 및 챗봇 응답 처리
if user_query := st.chat_input(config["placeholder"]):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # LangChain용 대화 기록 변환
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # RAG 파이프라인 가동 및 응답 스트리밍 출력
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("위키 DB 검색 중..."):
            # 검색은 현재 질문으로만 수행하여 주제 전환 대응
            # 대명사 생략 질문은 LLM이 대화 기록을 읽고 문맥 파악
            retrieved_docs = vectorstore.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "관련 정보를 찾을 수 없습니다."
            
            chain = qa_prompt | llm
            for chunk in chain.stream({
                "context": context_text,
                "chat_history": chat_history,
                "input": user_query
            }):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
