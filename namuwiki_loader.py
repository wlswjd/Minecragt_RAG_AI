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
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

namuwiki_urls = [
    ("마인크래프트(나무위키)", "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8"),
    ("마인크래프트/팁(나무위키)", "https://namu.wiki/w/%EB%A7%88%EC%9D%B8%ED%81%AC%EB%9E%98%ED%94%84%ED%8A%B8/%ED%8C%81")
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

total_added = 0

print("\n=== 나무위키 데이터 수집 시작 ===")
for title, url in namuwiki_urls:
    print(f"[{title}] 파싱 중... ({url})")
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 나무위키 본문 추출 (가장 텍스트가 많은 div를 찾거나 전체 텍스트 추출)
        # 나무위키는 구조가 복잡하므로, 불필요한 태그(script, style) 제거 후 텍스트 추출
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator='\n', strip=True)
        
        # 너무 짧은 줄이나 불필요한 공백 제거
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        clean_text = "\n".join(lines)
        
        chunks = text_splitter.split_text(clean_text)
        
        documents = []
        for chunk in chunks:
            content = f"출처: {title}\n내용: {chunk}"
            # 메타데이터에 source를 namuwiki로 명시하여 나중에 구분 가능하게 함
            documents.append(Document(page_content=content, metadata={"item": title, "type": "커뮤니티팁", "source": "namuwiki"}))
            
        if documents:
            vectorstore.add_documents(documents)
            total_added += len(documents)
            print(f"  -> 성공! {len(documents)}개의 데이터 청크 DB 추가 완료.")
        else:
            print("  -> 추출된 텍스트가 없습니다.")
            
    except Exception as e:
        print(f"  -> 에러 발생: {e}")
        
    time.sleep(2) # 서버 과부하 방지

print(f"\n--- 나무위키 수집 완료! 총 {total_added}개의 데이터가 추가되었습니다. ---")
