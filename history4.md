# Minecraft RAG AI 챗봇 개발 히스토리 (Part 4: RAG 고도화 — 메타데이터 보강 / HNSW / Hybrid Search / Re-ranking / 재인덱싱)

## 배경 및 목표
기존 RAG(history1~3)는 순수 Dense 검색(`similarity_search(k=5)`)만 사용했음.
RAG 학습(Day 1~6) 내용을 실제 프로젝트에 적용하여 검색 품질을 단계적으로 개선하는 작업 진행.
개선 계획 전체는 `CLAUDE.md`에 명시되어 있으며, 이번 파트에서 전 단계(메타데이터 보강 / HNSW / Hybrid Search / Re-ranking / 재인덱싱)를 모두 완료함.

---

## 1단계: 메타데이터 보강 (`batch_loader_full.py`)

### 배경
기존 Document 메타데이터가 `{"item": ..., "type": ...}` 두 필드뿐이었음.
향후 메타데이터 필터링 및 출처 구분을 위해 `source`, `category`, `url` 세 필드를 추가.

### 변경 내용
* `get_full_item_data()` 함수 시그니처에 `category=None`, `source="official_wiki"` 파라미터 추가.
* `base_meta` 딕셔너리로 공통 메타데이터를 일괄 관리:
  * `source`: 데이터 출처 구분 (`"official_wiki"` / `"namu_wiki"` / `"dcinside"`)
  * `category`: 수집 루프의 카테고리 변수를 그대로 전달 (예: `"도구"`, `"몹"`, `"core_system"`)
  * `url`: `urllib.parse.quote`로 한국어 인코딩된 위키 페이지 URL
* 단일 핵심 문서 루프: `category="core_system"` 고정.
* 카테고리 기반 루프: `category=category` (루프 변수 직접 전달).
* 정밀조합법 Document는 `output_item`(결과물)의 URL로 덮어씀.
* `from urllib.parse import quote` 추가.

### 주의
기존 `chroma_db`에 이미 저장된 데이터는 구 메타데이터 그대로.
신규 수집분부터만 보강된 메타데이터가 붙음.

---

## 2단계: HNSW 파라미터 명시 (`batch_loader_full.py`, `app.py`)

### 배경
LangChain Chroma의 기본값은 `hnsw:space=l2`이며, M/ef 파라미터도 암묵적으로 사용됨.
CLAUDE.md에서 결정한 값(`cosine`, `M=16`, `construction_ef=100`, `search_ef=30`)을 명시적으로 지정.

### 변경 내용
* `batch_loader_full.py`의 Chroma 인스턴스 생성에 `collection_metadata` 인자 추가.
* `app.py`의 `load_vectorstore()` 함수 내부에 `hnsw_meta` 딕셔너리를 정의하고 두 분기(collection_name 있음/없음) 모두에 적용.

### 실질적 효과 및 한계
* `normalize_embeddings=True` 덕분에 모든 벡터가 이미 단위 벡터 → L2와 cosine의 랭킹 결과가 수학적으로 동일. **cosine 변경의 실질 효과 없음.**
* `M=16`, `construction_ef=100`은 ChromaDB의 기본값과 동일. 변경 없음.
* `search_ef=30`은 기본(10)보다 높아 recall이 소폭 향상되나, 기존 컬렉션에는 소급 적용 불가.
* **결론:** 이 파라미터들은 다음 재인덱싱(청크 크기 400으로 축소) 시점에 함께 적용될 코드를 미리 맞춰둔 것임. 실제 효과는 재인덱싱 이후.

---

## 3단계: Hybrid Search 도입 (`hybrid_search.py` 신규, `app.py` 수정)

### 배경
순수 Dense 검색은 키워드 정확 매칭에 약함 (예: "철 곡괭이 내구도 몇?").
BM25는 키워드 검색에 강하지만 의미론적 유사도를 모름.
두 방식을 RRF(Reciprocal Rank Fusion)로 결합해 상호 보완.

### `hybrid_search.py` 구조

**`tokenize(text)`**
* 한국어 공백 기반 단순 토큰화 (정규식으로 특수문자 제거 후 split).
* 형태소 분석기(konlpy/Mecab) 없이 동작 — CPU 환경 호환.

**`build_bm25_index(vectorstore, bm25_path)`**
* `vectorstore._collection.get()`으로 ChromaDB 전체 청크를 한 번에 로드.
* `BM25Okapi(tokenized, k1=1.5, b=0.75)`로 인덱스 생성.
* `{"bm25", "ids", "documents", "metadatas"}` 딕셔너리를 pickle로 저장.
* 마인크래프트(`langchain` 컬렉션) → `bm25_minecraft.pkl`, 발헤임(`valheim`) → `bm25_valheim.pkl`로 분리.
* 데이터 추가(batch_loader 재실행) 후 반드시 재실행해야 인덱스 갱신됨.

**`load_bm25_index(bm25_path)`**
* pickle에서 `(bm25, ids, documents, metadatas)` 튜플로 로드.

**`hybrid_search(query, vectorstore, bm25_data, top_n=30, top_k=5)`**
1. Dense 검색: `vectorstore._embedding_function.embed_query(query)` → `vectorstore._collection.query()` (상위 `top_n`개)
2. BM25 검색: `bm25.get_scores(tokens)` → 점수 양수인 것만 상위 `top_n`개
3. RRF 결합: `score += 1 / (60 + rank)` 로 양쪽 순위 합산 후 정렬
4. 상위 `top_k`개를 LangChain `Document` 리스트로 변환하여 반환 (`app.py`와 바로 호환)

**`__main__` 블록**
* `python hybrid_search.py` 실행 시 두 컬렉션의 BM25 인덱스를 모두 빌드.
* 빈 컬렉션은 자동 스킵.

### `app.py` 변경 내용
* `from hybrid_search import hybrid_search as _hybrid_search, load_bm25_index, BM25_PATHS` 추가.
* `@st.cache_resource load_bm25(collection_name)` 함수 추가 — BM25 인덱스 파일이 있으면 로드, 없으면 `None` 반환.
* 검색 부분을 조건 분기로 교체:
  * BM25 인덱스 있음 → `_hybrid_search(user_query, vectorstore, bm25_data, top_n=30, top_k=5)`
  * BM25 인덱스 없음 → 기존 `vectorstore.similarity_search(user_query, k=5)` (폴백)
* 스피너 문구를 `"위키 DB 검색 중..."` → `"DB 하이브리드 검색 중..."` 으로 변경.

### 라이브러리 추가
* `rank-bm25` → `requirements.txt`에 추가 및 venv에 설치 완료.

---

---

## 4단계: Re-ranking 도입 (`reranker.py` 신규, `app.py` 수정)

### 배경
Hybrid Search가 RRF로 상위 30개 후보를 뽑더라도, 순위 기준이 여전히 Dense+BM25 점수의 합산임.
Cross-Encoder(재랭커)는 쿼리-문서 쌍을 직접 비교해 더 정밀한 점수를 매기므로 최종 5개의 품질이 올라감.

### `reranker.py` 구조
* `load_reranker()` — `Dongjin-kr/ko-reranker` CrossEncoder 로드 (`device="cpu"`).
* `rerank(query, docs, reranker, top_k=5)` — `(query, doc.page_content)` 쌍을 CrossEncoder에 넣어 점수순 정렬 후 상위 `top_k` 반환. 반환 타입은 LangChain `Document` 리스트.

### `app.py` 추가 변경
* `from reranker import load_reranker, rerank as _rerank` 추가.
* `@st.cache_resource load_reranker_model()` 함수 추가 — 앱 기동 시 1회 로드 후 캐싱.
* 검색 파이프라인 최종 형태:
  1. `_hybrid_search(..., top_n=30, top_k=30)` → 후보 30개
  2. `_rerank(query, candidates, reranker, top_k=5)` → 최종 5개
  3. BM25 인덱스 없는 경우 기존 `similarity_search(k=5)` 폴백 유지.

### 주의
* `Dongjin-kr/ko-reranker` 모델 크기 약 500MB — 앱 첫 실행(또는 Streamlit Cloud 재기동) 시 HuggingFace에서 자동 다운로드.
* Streamlit Community Cloud RAM 1GB 한계 상 임베딩 모델 + 재랭커 동시 상주가 부담될 수 있음. OOM 발생 시 재랭커 제외 분기 검토 필요.

---

## 5단계: 전체 재인덱싱

### 배경
* 기존 청크 크기(800~1000자)가 임베딩 모델(`ko-sroberta`, 토큰 한계 약 300~400자 한국어) 초과.
* 재인덱싱으로 청크 크기를 줄이고, HNSW 파라미터도 실제 컬렉션에 적용.

### 변경 내용
4개 로더 파일 청크 크기 전부 통일:

| 파일 | 변경 전 | 변경 후 |
|------|---------|---------|
| `batch_loader_full.py` | chunk_size=800, overlap=100 | chunk_size=400, overlap=80 |
| `namuwiki_loader.py` | chunk_size=1000, overlap=200 | chunk_size=400, overlap=80 |
| `valheim_namuwiki_loader.py` | chunk_size=1000, overlap=200 | chunk_size=400, overlap=80 |
| `valheim_dcinside_loader.py` | chunk_size=1000, overlap=200 | chunk_size=400, overlap=80 |

### 실행 순서
```bash
# 기존 DB·인덱스·진행 기록 전부 초기화
rm -rf chroma_db/ bm25_minecraft.pkl bm25_valheim.pkl
rm -f processed_items.txt processed_dc_posts.txt

# 마인크래프트 수집
python batch_loader_full.py
python namuwiki_loader.py

# 발헤임 수집
python valheim_namuwiki_loader.py
python valheim_dcinside_loader.py

# BM25 재빌드
python hybrid_search.py

# GitHub 푸시 (Streamlit Cloud 자동 반영)
git add chroma_db/ bm25_minecraft.pkl bm25_valheim.pkl processed_items.txt processed_dc_posts.txt
git commit -m "reindex: chunk_size 400/80 전체 통일, DB·BM25 재구축"
git push
```

### 효과
* 임베딩 품질 향상 (토큰 초과 구간 제거).
* HNSW `collection_metadata` 파라미터가 신규 컬렉션에 실제 적용.
* BM25 인덱스도 새 청크 기준으로 재구축.

---

## 최종 파이프라인 (완성)

```
사용자 질문
    ↓
Dense 검색 (top 30)  +  BM25 검색 (top 30)
    ↓
RRF 결합 → 후보 30개
    ↓
Cross-Encoder (Dongjin-kr/ko-reranker) → 최종 5개
    ↓
Gemini 2.5 Flash → 답변
```

## 최종 파일 구조

| 파일 | 역할 |
|------|------|
| `batch_loader_full.py` | 마인크래프트 공식 위키 크롤러 — 메타데이터(source/category/url), HNSW, chunk 400 |
| `namuwiki_loader.py` | 마인크래프트 나무위키 크롤러 — chunk 400 |
| `valheim_namuwiki_loader.py` | 발헤임 나무위키 크롤러 — chunk 400 |
| `valheim_dcinside_loader.py` | 발헤임 디시 갤러리 크롤러 — chunk 400 |
| `hybrid_search.py` | BM25 인덱스 빌드 + Hybrid Search 모듈 |
| `reranker.py` | Cross-Encoder Re-ranking 모듈 |
| `app.py` | Streamlit UI — 전체 파이프라인 연결 완료 |
