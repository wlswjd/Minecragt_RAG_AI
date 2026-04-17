# ⛏️ 마인크래프트 지능형 가이드 (Minecraft RAG AI)

마인크래프트 공식 위키와 커뮤니티의 방대한 데이터를 바탕으로, 게임 플레이에 필요한 모든 정보를 빠르고 정확하게 알려주는 지능형 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 🌟 주요 기능
* **정확한 공식 정보 제공:** 마인크래프트 공식 위키의 아이템, 몹, 생물 군계, 구조물, 마법 부여, 명령어 등 800여 개의 핵심 문서를 학습하여 정확한 스펙과 조합법을 안내합니다.
* **커뮤니티 꿀팁 & 글리치:** 나무위키 등에서 수집한 2,200여 개의 커뮤니티 팁, 스피드런 기술, 파밍 효율화 방법 등을 제공합니다. (비공식 정보의 경우 환각 방지를 위해 명확한 경고문구와 함께 제공됩니다.)
* **대화 문맥 인지 (Memory):** 이전 대화의 흐름을 기억합니다. 주어를 생략하고 "체력은 몇이야?", "어떻게 잡아?"라고 이어서 질문해도 찰떡같이 문맥을 파악하여 답변합니다.
* **실시간 스트리밍 답변:** ChatGPT처럼 답변이 실시간으로 타이핑되어 출력되므로 답답함 없이 쾌적하게 이용할 수 있습니다.

## 🛠️ 기술 스택 (Tech Stack)
* **Language:** Python 3.12
* **Frontend/UI:** Streamlit
* **LLM:** Google Gemini 2.5 Flash (`langchain-google-genai`)
* **Vector Database:** ChromaDB (`langchain-chroma`)
* **Embeddings:** HuggingFace `jhgan/ko-sroberta-multitask` (한국어 특화 임베딩)
* **Framework:** LangChain (RAG 파이프라인 및 History-Aware Retrieval 구현)
* **Data Scraping:** BeautifulSoup4, Requests (MediaWiki API 연동)

## 📂 프로젝트 구조
* `app.py`: Streamlit 웹 애플리케이션 및 RAG 챗봇 메인 실행 파일
* `batch_loader_full.py`: 마인크래프트 공식 위키 API 대량 크롤링 및 벡터 DB 적재 스크립트
* `namuwiki_loader.py`: 나무위키 심층 크롤링(Deep Crawl) 및 팁/글리치 데이터 적재 스크립트
* `chroma_db/`: 벡터화된 문서 청크(Chunk)들이 저장되는 로컬 데이터베이스 폴더
* `requirements.txt`: 프로젝트 실행에 필요한 Python 패키지 목록
* `history1~3.md`: 프로젝트 개발 및 트러블슈팅 과정 기록

## 🚀 실행 방법 (Local)
1. 저장소를 클론(Clone)합니다.
2. 가상환경을 생성하고 패키지를 설치합니다.
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```
3. `.env` 파일을 생성하고 Google Gemini API 키를 입력합니다.
   ```env
   GOOGLE_API_KEY="당신의_API_키"
   ```
4. Streamlit 앱을 실행합니다.
   ```bash
   streamlit run app.py
   ```

## 📝 평가 및 검증 (RAG Triad)
본 프로젝트는 RAG 시스템의 신뢰성을 위해 다음 기준을 고려하여 설계되었습니다.
1. **Context Relevance:** 사용자의 질문에 정확히 부합하는 문서를 검색하도록 쿼리 최적화 적용.
2. **Faithfulness:** DB에 없는 내용을 질문할 경우, 임의로 지어내지 않고 사전 지식을 사용하되 명확한 면책 조항(경고문)을 출력하도록 프롬프트 엔지니어링 적용.
3. **Answer Relevance:** 조합법, 몹 스펙 등 질문의 유형에 따라 가장 적합한 형태(마크다운, 3x3 그리드 설명 등)로 포맷팅하여 답변.
