import os
import re
import time
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("DB 및 임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Valheim 컬렉션에 함께 저장 (나무위키 데이터와 동일 컬렉션, source로 구분)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="valheim"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

BASE_URL = "https://gall.dcinside.com"
LIST_URL = "https://gall.dcinside.com/mgallery/board/lists/"
VIEW_URL = "https://gall.dcinside.com/mgallery/board/view/"

# 개념글 + 공략 카테고리 필터 (search_head=40 = 공략, exception_mode=recommend = 개념글)
LIST_PARAMS_BASE = {
    "id": "valheim",
    "sort_type": "N",
    "exception_mode": "recommend",
    "search_head": "40",
}

# 디시는 봇 차단이 엄격하므로 실제 브라우저 헤더를 최대한 모방
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://gall.dcinside.com/",
    "Connection": "keep-alive",
}

PROGRESS_FILE = "processed_dc_posts.txt"
# 개념글+공략 카테고리는 누적 약 31개로 1페이지에 전부 표시됨.
# 안전 차원에서 3페이지까지 시도하되, 같은 글 ID가 반복되면 자동 종료.
MAX_PAGES = 3
REQUEST_DELAY = 5  # 디시 IP 차단 방지 (초)


def load_processed_ids():
    """이미 처리한 글 번호 로드"""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_processed_id(post_id: str):
    """처리 완료한 글 번호 기록"""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{post_id}\n")


def get_post_ids_from_list(page: int) -> list[tuple[str, str]]:
    """리스트 페이지에서 (글번호, 제목) 튜플 목록 추출"""
    params = {**LIST_PARAMS_BASE, "page": str(page)}
    try:
        res = requests.get(LIST_URL, params=params, headers=headers, timeout=15)
        res.raise_for_status()
    except Exception as e:
        print(f"  리스트 페이지 {page} 요청 실패: {e}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    posts = []

    # 글 행: tr.us-post (디시 마이너 갤러리 표준)
    for tr in soup.select("tr.us-post"):
        # 공지/AD 행 스킵
        gubun = tr.select_one(".gall_subject")
        if gubun and gubun.get_text(strip=True) in ("공지", "설문", "AD"):
            continue

        num_td = tr.select_one(".gall_num")
        title_a = tr.select_one(".gall_tit a")
        if not num_td or not title_a:
            continue

        post_id = num_td.get_text(strip=True)
        if not post_id.isdigit():
            continue

        title = title_a.get_text(strip=True)
        # 댓글 수 표기 [N] 제거
        title = re.sub(r"\[\d+\]$", "", title).strip()
        posts.append((post_id, title))

    return posts


def extract_post_content(post_id: str) -> tuple[str, str] | None:
    """개별 글 페이지에서 (제목, 본문) 추출. 본문이 너무 짧으면 None"""
    params = {
        "id": "valheim",
        "no": post_id,
        "exception_mode": "recommend",
        "search_head": "40",
        "page": "1",
    }
    try:
        res = requests.get(VIEW_URL, params=params, headers=headers, timeout=15)
        res.raise_for_status()
    except Exception as e:
        print(f"  글 {post_id} 요청 실패: {e}")
        return None

    soup = BeautifulSoup(res.text, "html.parser")

    # 제목
    title_tag = soup.select_one(".title_subject") or soup.select_one(".tit_view")
    title = title_tag.get_text(strip=True) if title_tag else f"글 {post_id}"

    # 본문 영역
    write_div = soup.select_one(".write_div")
    if not write_div:
        return None

    # 노이즈 제거 (스크립트/스타일/이미지 caption 등)
    for tag in write_div(["script", "style", "iframe"]):
        tag.decompose()

    text = write_div.get_text(separator="\n", strip=True)
    # 빈 줄 정리
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 1]
    clean_text = "\n".join(lines)

    # 이미지 위주 글(텍스트 100자 미만)은 RAG에 부적합하므로 스킵
    if len(clean_text) < 100:
        return None

    return title, clean_text


def crawl_dcinside():
    processed = load_processed_ids()
    print(f"이미 처리된 글: {len(processed)}개")

    total_added = 0
    total_posts = 0
    skipped_empty = 0
    seen_post_ids = set()  # 페이지 반복 감지용

    for page in range(1, MAX_PAGES + 1):
        print(f"\n=== 리스트 페이지 {page} 수집 중 ===")
        posts = get_post_ids_from_list(page)
        if not posts:
            print(f"  페이지 {page}에서 글을 찾지 못함. 크롤링 종료.")
            break

        # 모든 글이 이전 페이지와 동일하면 마지막 페이지로 간주하고 종료
        new_ids_in_page = {pid for pid, _ in posts} - seen_post_ids
        if not new_ids_in_page and page > 1:
            print(f"  페이지 {page}는 이전과 동일. 마지막 페이지 도달, 종료.")
            break
        seen_post_ids.update(pid for pid, _ in posts)

        print(f"  페이지 {page}에서 {len(posts)}개 글 발견 (신규 {len(new_ids_in_page)}개)")
        time.sleep(REQUEST_DELAY)

        for post_id, list_title in posts:
            if post_id in processed:
                print(f"  [스킵] 이미 처리됨: {post_id} - {list_title}")
                continue

            print(f"  [수집] {post_id}: {list_title}")
            result = extract_post_content(post_id)

            if result is None:
                print(f"    -> 본문 없음/이미지 위주. 스킵.")
                skipped_empty += 1
                save_processed_id(post_id)  # 다음 실행 시 또 시도하지 않도록 기록
                time.sleep(REQUEST_DELAY)
                continue

            title, content = result
            chunks = text_splitter.split_text(content)
            documents = [
                Document(
                    page_content=f"출처: 디시인사이드 발헤임 갤러리 공략 - {title}\n내용: {chunk}",
                    metadata={
                        "item": title,
                        "type": "발헤임커뮤니티공략",
                        "source": "dcinside",
                        "game": "valheim",
                        "post_id": post_id,
                    },
                )
                for chunk in chunks
            ]

            if documents:
                vectorstore.add_documents(documents)
                total_added += len(documents)
                total_posts += 1
                print(f"    -> 성공! {len(documents)}개 청크 DB 추가")

            save_processed_id(post_id)
            processed.add(post_id)
            time.sleep(REQUEST_DELAY)

    print("\n=== 크롤링 완료 ===")
    print(f"신규 수집 글: {total_posts}개")
    print(f"본문 부족 스킵: {skipped_empty}개")
    print(f"DB에 추가된 청크: 총 {total_added}개")


if __name__ == "__main__":
    print("\n=== 디시인사이드 발헤임 갤러리 [개념글+공략] 수집 시작 ===")
    print(f"요청 간격: {REQUEST_DELAY}초 / 최대 페이지: {MAX_PAGES}\n")
    crawl_dcinside()
