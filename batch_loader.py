import time
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 수집 대상 아이템 리스트 정의 (기본 도구 및 무기류)
target_items = ["검", "도끼", "삽", "괭이", "방패", "활"]

# 2. 로컬 임베딩 모델 및 기존 벡터 DB 로드
print("--- 환경 초기화 중 ---")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
persist_dir = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 3. 데이터 수집 및 파싱 함수
def fetch_and_parse_recipe(item_name):
    api_url = "https://ko.minecraft.wiki/api.php"
    params = {"action": "parse", "page": item_name, "format": "json", "prop": "text"}
    headers = {"User-Agent": "MinecraftRAGProject/1.0"}

    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # 문서가 존재하지 않는 경우 예외 처리
        if "error" in data:
            print(f"[{item_name}] 문서 검색 실패: {data['error'].get('info', 'Unknown error')}")
            return []

        html_content = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html_content, 'html.parser')
        crafting_table = soup.find('table', attrs={'data-description': '제작법'})
        
        documents = []
        if crafting_table:
            for row in crafting_table.find_all('tr'):
                cols = [col.get_text(separator=' ', strip=True) for col in row.find_all(['th', 'td'])]
                if cols and len(cols) > 1 and cols[0] != '재료':
                    materials = cols[0]
                    description = cols[2].replace('\u200c', '').strip() if len(cols) > 2 else ""
                    
                    content_text = f"아이템: {item_name}\n조합 재료: {materials}\n추가 설명: {description}"
                    meta_info = {"source": "ko.minecraft.wiki", "category": "장비", "type": "조합법", "item": item_name}
                    
                    doc = Document(page_content=content_text.strip(), metadata=meta_info)
                    documents.append(doc)
            print(f"[{item_name}] 조합법 데이터 {len(documents)}건 추출 완료.")
        else:
            print(f"[{item_name}] 문서 내 '제작법' 표를 찾을 수 없습니다.")
        
        return documents

    except Exception as e:
        print(f"[{item_name}] 처리 중 에러 발생: {e}")
        return []

# 4. 일괄 처리(Batch Processing) 실행
all_new_documents = []

print("\n--- 데이터 일괄 수집 시작 ---")
for item in target_items:
    docs = fetch_and_parse_recipe(item)
    if docs:
        all_new_documents.extend(docs)
    
    # 위키 서버 부하 방지를 위한 1초 대기
    time.sleep(1)

# 5. 수집된 데이터를 벡터 DB에 추가
if all_new_documents:
    print(f"\n--- 벡터 DB 업데이트 시작 (총 {len(all_new_documents)}건) ---")
    vectorstore.add_documents(documents=all_new_documents)
    print("벡터 DB 업데이트 완료.")
else:
    print("\n업데이트할 새로운 데이터가 없습니다.")