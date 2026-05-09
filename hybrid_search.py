import re
import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

RRF_K = 60

# 컬렉션별 BM25 인덱스 경로
BM25_PATHS = {
    "langchain": "./bm25_minecraft.pkl",
    "valheim": "./bm25_valheim.pkl",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def build_bm25_index(vectorstore, bm25_path: str, collection_name: str = ""):
    """ChromaDB의 모든 청크를 가져와 BM25 인덱스 생성 후 pickle로 저장.
    새 데이터를 batch_loader로 추가한 뒤 반드시 재실행해야 인덱스가 갱신됨.
    """
    label = collection_name or bm25_path
    print(f"[{label}] ChromaDB 청크 로딩 중...")

    result = vectorstore.get(include=["documents", "metadatas"])
    ids = result["ids"]
    documents = result["documents"]
    metadatas = result["metadatas"]

    print(f"  총 {len(documents)}개 청크 로드. BM25 인덱싱 중...")
    tokenized = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)

    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }, f)

    print(f"  저장 완료: {bm25_path} ({len(documents)}개 청크)\n")
    return bm25, ids, documents, metadatas


def load_bm25_index(bm25_path: str):
    """저장된 BM25 인덱스를 로드해 (bm25, ids, documents, metadatas) 튜플 반환."""
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"], data["documents"], data["metadatas"]


def _rrf_combine(dense_results, bm25_results, rrf_k: int = RRF_K) -> list[str]:
    """두 검색 결과 (id, rank) 리스트를 RRF로 결합해 id 순서 반환."""
    scores: dict[str, float] = {}
    for doc_id, rank in dense_results:
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
    for doc_id, rank in bm25_results:
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def hybrid_search(
    query: str,
    vectorstore,
    bm25_data: tuple,
    top_n: int = 30,
    top_k: int = 5,
) -> list[Document]:
    """BM25 + Dense → RRF 결합 → LangChain Document 리스트 반환.

    Args:
        query: 사용자 질문
        vectorstore: LangChain Chroma 인스턴스
        bm25_data: load_bm25_index() 가 반환한 (bm25, ids, documents, metadatas) 튜플
        top_n: 각 검색기에서 후보로 뽑을 수 (CLAUDE.md: 30)
        top_k: 최종 반환 문서 수 (CLAUDE.md: 5)
    """
    bm25, ids, documents, metadatas = bm25_data
    idx_to_id = {i: doc_id for i, doc_id in enumerate(ids)}
    id_to_doc = {doc_id: (documents[i], metadatas[i]) for i, doc_id in enumerate(ids)}
    # Dense 결과(Document)를 BM25 ID와 연결하기 위한 역매핑
    content_to_id = {doc: doc_id for doc_id, (doc, _) in id_to_doc.items()}

    # 1. Dense 검색 — LangChain 공개 API만 사용
    dense_docs = vectorstore.similarity_search(query, k=top_n)
    dense_results = []
    for rank, doc in enumerate(dense_docs):
        doc_id = content_to_id.get(doc.page_content)
        if doc_id:
            dense_results.append((doc_id, rank + 1))

    # 2. BM25 검색
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    scored = sorted(
        [(idx_to_id[i], s) for i, s in enumerate(bm25_scores) if s > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    bm25_results = [
        (doc_id, rank + 1)
        for rank, (doc_id, _) in enumerate(scored[:top_n])
    ]

    # 3. RRF 결합
    merged_ids = _rrf_combine(dense_results, bm25_results)[:top_k]

    # 4. LangChain Document로 변환
    return [
        Document(page_content=id_to_doc[doc_id][0], metadata=id_to_doc[doc_id][1])
        for doc_id in merged_ids
        if doc_id in id_to_doc
    ]


if __name__ == "__main__":
    # BM25 인덱스 빌드 — 처음 한 번, 또는 데이터 추가 후 재실행
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    for collection_name, bm25_path in BM25_PATHS.items():
        vs = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        result = vs.get(include=["documents"])
        if not result["ids"]:
            print(f"[{collection_name}] 컬렉션이 비어 있음. 스킵.\n")
            continue
        build_bm25_index(vs, bm25_path, collection_name)

    print("=== 전체 BM25 인덱스 빌드 완료 ===")
