import os
from dotenv import load_dotenv
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

# 1. 환경 변수 로드
load_dotenv()

# 2. USER_AGENT 환경 변수 직접 설정 (경고 메시지 해결)
os.environ["USER_AGENT"] = "MinecraftRAGProject/1.0"

# 3. 타겟 URL 설정
url = "https://minecraft.wiki/w/Pickaxe"

# 4. 파싱 규칙 설정: 위키 본문 영역(mw-parser-output)만 추출하도록 지정
bs_kwargs = {
    "parse_only": SoupStrainer("div", class_="mw-parser-output")
}

# 5. 지정된 파싱 규칙을 적용하여 로더 실행
loader = WebBaseLoader(web_paths=[url], bs_kwargs=bs_kwargs)
docs = loader.load()

# 6. 수집 결과 확인
print(f"수집된 문서 개수: {len(docs)}")
print("--- 정제된 문서 내용 미리보기 (최초 500자) ---")
print(docs[0].page_content[:500].strip())