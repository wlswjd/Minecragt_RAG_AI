import os
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
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# 이미 처리한 아이템을 기록할 파일
PROCESSED_FILE = "processed_items.txt"

def load_processed_items():
    """메모장 파일에서 이미 수집 완료한 아이템 목록을 불러옵니다."""
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def save_processed_item(item_name):
    """수집 완료한 아이템을 메모장 파일에 추가합니다."""
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{item_name}\n")

def get_all_items_from_category(category_name, limit=None, visited_categories=None):
    """
    위키 API를 사용하여 특정 카테고리의 모든 문서 제목 목록을 가져옵니다.
    하위 카테고리(ns=14)가 발견되면 재귀적으로 탐색하여 모든 문서를 수집합니다.
    """
    if visited_categories is None:
        visited_categories = set()
        
    # 이미 탐색한 카테고리면 무한 루프 방지를 위해 스킵
    if category_name in visited_categories:
        return []
    visited_categories.add(category_name)
    
    api_url = "https://ko.minecraft.wiki/api.php"
    items = []
    cmcontinue = None
    headers = {"User-Agent": "MinecraftRAGProject/1.0 (Contact: myemail@example.com)"}
    
    print(f"[{category_name}] 카테고리(및 하위) 탐색 중...")
    
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_name}",
            "cmlimit": "max" if limit is None else limit,
            "format": "json"
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
            
        try:
            res = requests.get(api_url, params=params, headers=headers)
            data = res.json()
            members = data.get("query", {}).get("categorymembers", [])
            
            for member in members:
                if member["ns"] == 0: # 일반 문서
                    items.append(member["title"])
                elif member["ns"] == 14: # 하위 카테고리
                    sub_category = member["title"].replace("Category:", "").replace("분류:", "")
                    # 재귀적으로 하위 카테고리 탐색
                    items.extend(get_all_items_from_category(sub_category, limit, visited_categories))
            
            if limit is not None or "continue" not in data or "cmcontinue" not in data["continue"]:
                break
                
            cmcontinue = data["continue"]["cmcontinue"]
            time.sleep(0.5)
            
        except Exception as e:
            print(f"카테고리 목록 가져오기 실패: {e}")
            break
            
    # 중복 제거 후 반환
    return list(set(items))

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
    # 1. 단일 핵심 문서 (시스템/메커니즘)
    target_pages = [
        "거래", "양조", "마법 부여", "제작", "제련", "대장장이 작업", 
        "레드스톤", "명령어", "발전 과제", "튜토리얼",
        # 패치노트/업데이트 내역 추가
        "버전 역사", "Java Edition 버전 역사", "Bedrock Edition 버전 역사", "계획된 버전"
    ]
    
    # 2. 여러 문서가 포함된 카테고리 (기존 아이템 외 추가)
    target_categories = [
        # 기존 아이템 관련
        "도구", "무기", "갑옷", "음식", "광석", "블록", "재료", "물약",
        # 신규 추가 (몹, 환경, 기타)
        "몹", "생물 군계", "효과", "구조물", "Structures", "Generated_structures",
        # 세부 시스템 하위 문서들 (인챈트 종류, 레드스톤 부품, 명령어 종류)
        "마법 부여", "레드스톤", "명령어"
    ]
    
    # 이미 수집한 아이템 목록 불러오기
    processed_items = load_processed_items()
    print(f"현재까지 수집 완료된 문서 수: {len(processed_items)}개")
    
    total_added = 0
    
    # --- 1. 단일 핵심 문서 수집 ---
    print("\n=== [1단계] 핵심 시스템/메커니즘 문서 수집 시작 ===")
    new_pages = [page for page in target_pages if page not in processed_items]
    print(f"수집할 핵심 문서: {len(new_pages)}개\n")

    for page in new_pages:
        print(f"[{page}] 파싱 중...")
        docs = get_full_item_data(page)
        
        if docs:
            vectorstore.add_documents(docs)
            total_added += len(docs)
            print(f"  -> 성공! {len(docs)}개의 데이터 DB 추가 완료.")
        else:
            print(f"  -> 파싱 실패 또는 데이터 없음")
            
        save_processed_item(page)
        processed_items.add(page)
        time.sleep(1)

    # --- 2. 카테고리 기반 문서 수집 ---
    print("\n=== [2단계] 카테고리 기반 문서 수집 시작 ===")
    for category in target_categories:
        print(f"\n--- '{category}' 카테고리 수집 시작 ---")
        
        # limit=None으로 설정하면 카테고리의 끝까지 모든 아이템 제목을 가져옵니다.
        items_to_fetch = get_all_items_from_category(category, limit=None)
        
        # 중복 제거: 이미 처리한 아이템은 목록에서 제외
        new_items = [item for item in items_to_fetch if item not in processed_items]
        
        print(f"카테고리 총 문서: {len(items_to_fetch)}개 | 새로 수집할 문서: {len(new_items)}개\n")
        
        for item in new_items:
            print(f"[{item}] 파싱 중...")
            docs = get_full_item_data(item)
            
            if docs:
                vectorstore.add_documents(docs)
                total_added += len(docs)
                print(f"  -> 성공! {len(docs)}개의 데이터 DB 추가 완료.")
            else:
                print(f"  -> 파싱 실패 또는 데이터 없음")
                
            # 성공하든 실패하든, 다시 시도하지 않도록 기록에 추가
            save_processed_item(item)
            processed_items.add(item)
            
            # 서버 과부하 방지를 위해 1초 대기
            time.sleep(1)
            
    print(f"\n--- 자동화 수집 완료! 이번 실행으로 총 {total_added}개의 데이터가 추가되었습니다. ---")
