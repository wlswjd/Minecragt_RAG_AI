from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

RERANKER_MODEL = "Dongjin-kr/ko-reranker"


def load_reranker() -> CrossEncoder:
    """ko-reranker Cross-Encoder 로드 (최초 실행 시 HuggingFace에서 다운로드)."""
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def rerank(query: str, docs: list[Document], reranker: CrossEncoder, top_k: int = 5) -> list[Document]:
    """Cross-Encoder로 후보 문서를 재정렬해 top_k개 반환."""
    if not docs:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]
