import time
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 환경 초기화
print("DB 및 임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 기존 DB가 삭제되었으므로 새로 생성됩니다.
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 일반 텍스트가 너무 길면 AI가 헷갈릴 수 있으므로 잘라서 넣습니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def get_items_from_category(category_name, limit=20):
    """위키 API를 사용하여 특정 카테고리의 문서 제목 목록을 가져옵니다."""
    api_url = "https://ko.minecraft.wiki/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": limit,
        "format": "json"
    }
    headers = {"User-Agent": "MinecraftRAGProject/1.0 (Contact: myemail@example.com)"}
    
    try:
        res = requests.get(api_url, params=params, headers=headers)
        data = res.json()
        items = [member["title"] for member in data.get("query", {}).get("categorymembers", [])]
        return items
    except Exception as e:
        print(f"카테고리 목록 가져오기 실패: {e}")
        return []

def get_full_item_data(item_name):
    """단일 아이템의 '일반 정보(텍스트)'와 '정밀 조합법'을 모두 파싱합니다."""
    api_url = "https://ko.minecraft.wiki/api.php"
    params = {"action": "parse", "page": item_name, "format": "json", "prop": "text"}
    headers = {"User-Agent": "MinecraftRAGProject/1.0 (Contact: myemail@example.com)"}

    try:
        res = requests.get(api_url, params=params, headers=headers)
        data = res.json()
        if "error" in data:
            return []
            
        html = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html, 'html.parser')
        documents = []
        
        # 1. 일반 텍스트 정보 추출 (p 태그)
        paragraphs = soup.find_all('p')
        general_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        if general_text:
            chunks = text_splitter.split_text(general_text)
            for chunk in chunks:
                content = f"아이템: {item_name}\n설명: {chunk}"
                documents.append(Document(page_content=content, metadata={"item": item_name, "type": "일반정보"}))

        # 2. 정밀 조합법 추출 (mcui-Crafting_Table)
        mcui_layouts = soup.find_all('span', class_='mcui-Crafting_Table')
        for layout in mcui_layouts:
            rows = layout.find_all('span', class_='mcui-row')
            slots = []
            for row in rows:
                slots.extend(row.find_all('span', class_='invslot'))
            
            output_span = layout.find('span', class_='mcui-output')
            output_item = item_name
            if output_span:
                output_a = output_span.find('a')
                if output_a and output_a.get('title'):
                    output_item = output_a.get('title')
            
            if len(slots) >= 9:
                grid = []
                empty_count = 0
                for i, slot in enumerate(slots[:9]):
                    item_span = slot.find('span', class_='invslot-item')
                    item = "빈칸"
                    if item_span:
                        a_tag = item_span.find('a')
                        if a_tag and a_tag.get('title'):
                            item = a_tag.get('title')
                    if item == "빈칸":
                        empty_count += 1
                    grid.append(f"{i+1}번칸:{item}")
                
                if empty_count == 9:
                    continue
                
                grid_display = f"\n[{grid[0]}][{grid[1]}][{grid[2]}]\n[{grid[3]}][{grid[4]}][{grid[5]}]\n[{grid[6]}][{grid[7]}][{grid[8]}]"
                content = f"아이템: {output_item}\n제작 배치도:{grid_display}\n재료 요약: {item_name} 제작용"
                documents.append(Document(page_content=content, metadata={"item": output_item, "type": "정밀조합법"}))
        
        return documents
    except Exception as e:
        print(f"Error parsing {item_name}: {e}")
        return []

if __name__ == "__main__":
    target_category = "도구"
    fetch_limit = 20
    
    print(f"\n--- '{target_category}' 카테고리 아이템 최대 {fetch_limit}개 수집 시작 (일반 정보 + 조합법) ---")
    items_to_fetch = get_items_from_category(target_category, limit=fetch_limit)
    
    # 방패도 명시적으로 포함 (테스트용)
    if "방패" not in items_to_fetch:
        items_to_fetch.append("방패")
        
    print(f"수집 대상 목록: {items_to_fetch}\n")
    
    total_added = 0
    for item in items_to_fetch:
        print(f"[{item}] 파싱 중...")
        docs = get_full_item_data(item)
        
        if docs:
            vectorstore.add_documents(docs)
            total_added += len(docs)
            print(f"  -> 성공! {len(docs)}개의 데이터(일반+조합법) DB 추가 완료.")
        else:
            print(f"  -> 파싱 실패 또는 데이터 없음")
            
        # 서버 과부하 방지를 위해 1초 대기
        time.sleep(1)
        
    print(f"\n--- 자동화 수집 완료! 총 {total_added}개의 데이터가 추가되었습니다. ---")
