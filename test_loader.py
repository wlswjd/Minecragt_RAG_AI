import requests
from bs4 import BeautifulSoup

# 1. API 요청 설정
api_url = "https://minecraft.wiki/api.php"
params = {
    "action": "parse",
    "page": "Pickaxe",
    "format": "json",
    "prop": "text"
}
headers = {"User-Agent": "MinecraftRAGProject/1.0"}

# 2. 데이터 호출
response = requests.get(api_url, params=params, headers=headers)
response.raise_for_status()
data = response.json()
html_content = data["parse"]["text"]["*"]

# 3. HTML 파싱
soup = BeautifulSoup(html_content, 'html.parser')

# 4. 문서 내의 모든 h2, h3 제목 태그 추출
print("--- 문서 내 존재하는 목차(Heading) 구조 ---")
for heading in soup.find_all(['h2', 'h3']):
    # MediaWiki의 제목은 보통 <span class="mw-headline"> 안에 들어있습니다.
    headline = heading.find('span', class_='mw-headline')
    if headline:
        print(f"[{heading.name}] {headline.text}")