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
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 시작점 URL 목록 (사진에서 요청하신 '인게임' 및 '도움말' 관련 메인 문서들)
start_urls = [
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EC%95%84%EC%9D%B4%ED%85%9C",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EB%AA%B9",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EC%83%9D%EB%AC%BC_%EA%B5%B0%EA%B3%84",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EA%B5%AC%EC%A1%B0%EB%AC%BC",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EB%A7%88%EB%B2%95_%EB%B6%80%EC%97%AC",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EC%83%81%ED%83%9C_%ED%9A%A8%EA%B3%BC",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%ED%8C%81",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EC%8A%A4%ED%84%B0%EB%93%9C%EB%9F%B0",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/PVP",
    "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%EA%B1%B4%EC%B6%95"
]

visited_urls = set()
total_added = 0

def crawl_namuwiki(url, current_depth, max_depth=1):
    global total_added
    
    # URL 디코딩 (한글 깨짐 방지 및 중복 체크용)
    decoded_url = unquote(url)
    
    # 앵커(#) 제거 (같은 문서의 다른 문단 중복 수집 방지)
    decoded_url = decoded_url.split('#')[0]
    
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
        
        # 본문 텍스트 추출 (불필요한 태그 제거)
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
            documents.append(Document(page_content=content, metadata={"item": title, "type": "커뮤니티팁", "source": "namuwiki"}))
            
        if documents:
            vectorstore.add_documents(documents)
            total_added += len(documents)
            print(f"  -> 성공! {len(documents)}개의 데이터 청크 DB 추가 완료.")
            
        # 하위 링크 탐색 (재귀 호출)
        if current_depth < max_depth:
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # 마인크래프트 관련 문서만 타겟팅 (무한 루프 및 엉뚱한 문서 방지)
                if href.startswith('/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8') or href.startswith('/w/마인크래프트'):
                    next_url = urljoin("https://namu.wiki", href)
                    crawl_namuwiki(next_url, current_depth + 1, max_depth)
                    
    except Exception as e:
        print(f"  -> 에러 발생: {e}")
        
    # [매우 중요] 나무위키 IP 차단 방지를 위한 필수 대기 시간
    time.sleep(3)

print("\n=== 나무위키 심층 데이터 수집 시작 ===")
for start_url in start_urls:
    # max_depth=1: 시작 문서 + 그 문서에 링크된 하위 문서(1단계)까지 탐색
    crawl_namuwiki(start_url, current_depth=0, max_depth=1)

print(f"\n--- 나무위키 수집 완료! 총 {total_added}개의 데이터가 추가되었습니다. ---")