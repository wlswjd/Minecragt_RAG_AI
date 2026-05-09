# 마크 RAG (Minecragt_RAG_AI) — 작업 컨텍스트

## 프로젝트 개요
한국어 마인크래프트 위키 + 커뮤니티 데이터 기반 RAG 챗봇.
- 데이터 수집: `batch_loader_full.py` (위키 API 크롤링)
- 인덱싱: ChromaDB persistent (`./chroma_db`)
- 검색 + LLM: `app.py` (Streamlit + LangChain + Gemini 2.5 Flash)
- 임베딩 모델: `jhgan/ko-sroberta-multitask`

## 현재 검색 구조
```python
# app.py의 현재 검색 부분
retrieved_docs = vectorstore.similarity_search(user_query, k=5)
```
→ 순수 Dense 검색만 사용. Hybrid/Re-ranking/메타데이터 필터 미적용.

## 적용할 개선사항 (RAG 학습 Day 1~6 기반)

### 1. 청킹 크기 검토
- 현재: chunk_size=800, overlap=100 (RecursiveCharacterTextSplitter)
- 문제: ko-sroberta 토큰 한계(약 300~400자 한국어) 초과 가능성
- 개선: chunk_size=400, overlap=80으로 축소 검토
- 단, 기존 인덱스 재구축이 필요해서 우선순위 조정 가능

### 2. HNSW 파라미터 명시
- 현재: LangChain Chroma 래퍼의 기본값 사용 (l2 가능성)
- 개선: cosine + M=16 + construction_ef=100 + search_ef=30 명시

### 3. Hybrid Search 도입 (BM25 + Dense + RRF)
- 라이브러리: `rank_bm25`
- 토큰화: 1차로 단순 정규식, 가능하면 konlpy/Mecab
- RRF k=60
- BM25 인덱스는 별도 pickle 파일로 관리 권장

### 4. Re-ranking 도입
- 모델: `Dongjin-kr/ko-reranker` (CPU 환경 적합)
- 파라미터: Top-N=30 → Top-K=5
- `sentence-transformers`의 `CrossEncoder` 사용

### 5. 메타데이터 필터링
- 현재 메타데이터: `item`, `type`만 있음
- 추가 권장: `source` (official_wiki/namu_wiki/community), `category`, `url`
- 검색 시 LangChain Chroma의 `filter` 파라미터 활용

## 결정된 파라미터
- 청크: chunk_size=400, overlap=80 (재인덱싱 시)
- HNSW: space=cosine, M=16, construction_ef=100, search_ef=30
- Hybrid: RRF k=60
- Re-rank: Top-N=30, Top-K=5
- 모델: jhgan/ko-sroberta-multitask (임베딩), Dongjin-kr/ko-reranker (재정렬)

## 작업 순서 권장
1. 메타데이터 보강 (batch_loader_full.py 수정) — 신규 데이터부터 적용
2. HNSW 파라미터 명시 (LangChain Chroma 생성 시 collection_metadata 인자)
3. Hybrid Search 모듈 분리 (`hybrid_search.py` 신규 파일 권장)
4. Re-ranking 모듈 분리 (`reranker.py` 신규 파일 권장)
5. app.py 검색 부분 교체

## 주의사항
- 기존 chroma_db는 재인덱싱하지 않으면 청크 크기/메타데이터 그대로
- BM25 인덱스 신규 생성 필요 (기존 청크들에 대해)
- LangChain Chroma 사용 중이므로 raw chromadb 직접 사용보다 LangChain wrapper 우선 시도
- `device='cpu'` 명시되어 있음 — Cross-encoder도 CPU에서 동작 확인 필요

## 참고 미니 프로젝트
RAG 학습 중 만든 미니 프로젝트(`rag-study/day7_mini/`)에 모든 기법이 통합된 예시 코드가 있음. 구조 참고 가능.