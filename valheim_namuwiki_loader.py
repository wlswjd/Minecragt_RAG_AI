import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote, urljoin
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

# Valheim 데이터는 별도 컬렉션(valheim)에 저장하여 마인크래프트 데이터와 분리
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="valheim"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 발헤임 위키 시작점 URL (메인 페이지에서 모든 하위 문서를 재귀 탐색)
start_urls = [
    "https://namu.wiki/w/Valheim",
]

visited_urls = set()
total_added = 0

def crawl_namuwiki(url, current_depth, max_depth=1):
    global total_added
    decoded_url = unquote(url).split('#')[0]
    
    if current_depth > max_depth:
        return
    if decoded_url in visited_urls:
        return
    visited_urls.add(decoded_url)
    print(f"[{current_depth}/{max_depth}] 파싱 중... {decoded_url}")
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        clean_text = "\n".join(lines)
        
        chunks = text_splitter.split_text(clean_text)
        documents = []
        title = decoded_url.split('/')[-1]
        
        for chunk in chunks:
            content = f"출처: {title}\n내용: {chunk}"
            documents.append(Document(
                page_content=content,
                metadata={"item": title, "type": "발헤임위키", "source": "namuwiki", "game": "valheim"}
            ))
            
        if documents:
            vectorstore.add_documents(documents)
            total_added += len(documents)
            print(f"  -> 성공! {len(documents)}개의 데이터 청크 DB 추가 완료.")
            
        # 하위 링크 재귀 탐색 (Valheim 관련 문서로만 한정)
        if current_depth < max_depth:
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith('/w/Valheim') or href.startswith('/w/%EB%B0%9C%ED%97%A4%EC%9E%84'):
                    next_url = urljoin("https://namu.wiki", href)
                    crawl_namuwiki(next_url, current_depth + 1, max_depth)
                    
    except Exception as e:
        print(f"  -> 에러 발생: {e}")
        
    # 나무위키 IP 차단 방지용 필수 대기
    time.sleep(3)

print("\n=== 발헤임 나무위키 심층 데이터 수집 시작 ===")
for start_url in start_urls:
    crawl_namuwiki(start_url, current_depth=0, max_depth=1)

print(f"\n--- 발헤임 수집 완료! 총 {total_added}개의 데이터가 추가되었습니다. ---")
