import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 환경 변수 로드 (.env 파일의 GOOGLE_API_KEY 인식)
load_dotenv()

# 2. API 요청 및 HTML 파싱
api_url = "https://ko.minecraft.wiki/api.php"
params = {"action": "parse", "page": "곡괭이", "format": "json", "prop": "text"}
headers = {"User-Agent": "MinecraftRAGProject/1.0"}

response = requests.get(api_url, params=params, headers=headers)
response.raise_for_status()
html_content = response.json()["parse"]["text"]["*"]
soup = BeautifulSoup(html_content, 'html.parser')

# 3. 데이터 구조화 및 LangChain Document 객체 변환
crafting_table = soup.find('table', attrs={'data-description': '제작법'})
langchain_documents = []

if crafting_table:
    for row in crafting_table.find_all('tr'):
        cols = [col.get_text(separator=' ', strip=True) for col in row.find_all(['th', 'td'])]
        if cols and len(cols) > 1 and cols[0] != '재료':
            materials = cols[0]
            description = cols[2].replace('\u200c', '').strip() if len(cols) > 2 else ""
            
            content_text = f"아이템: 곡괭이\n조합 재료: {materials}\n추가 설명: {description}"
            meta_info = {"source": "ko.minecraft.wiki", "category": "도구", "type": "조합법"}
            
            doc = Document(page_content=content_text.strip(), metadata=meta_info)
            langchain_documents.append(doc)

print(f"문서 {len(langchain_documents)}개 추출 및 객체 변환 완료.")

# 4. 임베딩 모델 설정 (Google Gemini API)
# models/embedding-001 모델을 사용하여 텍스트를 벡터로 변환합니다.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 5. Chroma 벡터 DB 구축 및 로컬 저장
print("\n--- 벡터 DB 구축 및 저장 시작 ---")
persist_dir = "./chroma_db"

# 기존에 동일한 이름의 DB가 있다면 덮어쓰거나 추가합니다.
vectorstore = Chroma.from_documents(
    documents=langchain_documents,
    embedding=embeddings,
    persist_directory=persist_dir
)
print(f"로컬 디렉토리('{persist_dir}')에 벡터 DB 저장 완료.")

# 6. 자연어 유사도 검색(Similarity Search) 테스트
query = "구리로 곡괭이를 어떻게 만들어?"
print(f"\n--- 검색 테스트 ---")
print(f"사용자 질문: {query}")

# k=1 파라미터는 질문과 의미적으로 가장 유사한 문서 1개만 가져오라는 의미입니다.
retrieved_docs = vectorstore.similarity_search(query, k=1)

for i, doc in enumerate(retrieved_docs):
    print(f"\n[검색된 문서 {i+1}]")
    print(doc.page_content)