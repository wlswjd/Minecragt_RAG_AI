from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

_RERANKER_MODEL = "Dongjin-kr/ko-reranker"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_RERANKER_MODEL, max_length=512, device="cpu")
    return _model


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Cross-Encoder로 후보 문서를 재정렬해 top_k개 반환."""
    if not docs:
        return docs
    model = _get_model()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]
