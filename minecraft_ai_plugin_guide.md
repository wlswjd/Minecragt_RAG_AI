# 🎮 마인크래프트 서버 AI 챗봇 연동 가이드 (Python RAG ↔ Java Plugin)

우리가 만든 파이썬 기반의 RAG AI를 마인크래프트 인게임 채팅창에서 사용하려면, 파이썬 코드를 직접 마인크래프트(자바)에 넣는 것이 아니라 **"클라이언트-서버(API) 아키텍처"**를 사용해야 합니다.

즉, **파이썬 AI는 뒤에서 조용히 API 서버로 돌아가고**, **마인크래프트 자바 플러그인이 채팅을 가로채서 파이썬 서버에 질문을 던지고 답변을 받아오는 방식**입니다.

---

## 🏗️ 전체 아키텍처 구조
1. **유저:** 인게임 채팅창에 `!ai 구리 곡괭이 어떻게 만들어?` 입력
2. **MC 플러그인 (Java):** 채팅을 가로채서 파이썬 서버로 HTTP POST 요청 전송
3. **AI 서버 (Python/FastAPI):** 요청을 받아 RAG DB를 검색하고 Gemini로 답변 생성 후 반환
4. **MC 플러그인 (Java):** 받은 답변을 마인크래프트 채팅창에 출력

---

## 🚀 1단계: 파이썬 AI를 API 서버로 만들기 (FastAPI)

기존 `app.py`는 웹(Streamlit)용이므로, 자바 플러그인과 통신하기 위해 **FastAPI**를 사용해 API 서버(`api.py`)를 새로 만듭니다.

### 1. 패키지 설치
터미널에서 FastAPI와 서버 구동기(Uvicorn)를 설치합니다.
```bash
./venv/bin/python -m pip install fastapi uvicorn pydantic
```

### 2. `api.py` 파일 작성
프로젝트 폴더에 `api.py`를 만들고 아래 코드를 작성합니다. (기존 RAG 로직을 그대로 가져옵니다.)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()

# 모델 및 DB 로드
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 마인크래프트 가이드입니다. 다음 Context를 기반으로 짧고 간결하게 대답하세요. 인게임 채팅창에 출력되므로 너무 길면 안 됩니다.\n\n[Context]\n{context}"),
    ("human", "{input}")
])

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_ai(req: AskRequest):
    # 1. DB 검색
    retrieved_docs = vectorstore.similarity_search(req.question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "정보 없음"
    
    # 2. 체인 실행
    chain = qa_prompt | llm
    response = chain.invoke({"context": context_text, "input": req.question})
    
    return {"answer": response.content}

# 서버 실행 명령어: ./venv/bin/python -m uvicorn api:app --reload
```

---

## 🚀 2단계: 마인크래프트 서버 및 플러그인 개발 환경 세팅

마인크래프트 플러그인을 개발하려면 **PaperMC** 서버와 **IntelliJ IDEA** (또는 Eclipse)가 필요합니다.

1. **PaperMC 서버 구동:** [PaperMC 공식 홈페이지](https://papermc.io/)에서 서버 버킷(.jar)을 다운받아 로컬에서 실행합니다.
2. **IDE 세팅:** IntelliJ IDEA를 설치하고, **Minecraft Development** 플러그인을 설치하면 프로젝트 생성이 매우 쉬워집니다.
3. 새 프로젝트 생성 시 `Paper Plugin`을 선택합니다.

---

## 🚀 3단계: 자바(Java) 플러그인 코드 작성

플러그인의 메인 클래스(예: `MinecraftRAGPlugin.java`)에 채팅을 감지하고 파이썬 서버와 통신하는 코드를 작성합니다.

### 1. 이벤트 리스너 작성 (채팅 가로채기)
```java
package com.yourname.minecraftrag;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.AsyncPlayerChatEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public final class MinecraftRAGPlugin extends JavaPlugin implements Listener {

    private final HttpClient httpClient = HttpClient.newHttpClient();

    @Override
    public void onEnable() {
        // 이벤트 리스너 등록
        getServer().getPluginManager().registerEvents(this, this);
        getLogger().info("RAG AI 플러그인이 활성화되었습니다!");
    }

    @EventHandler
    public void onPlayerChat(AsyncPlayerChatEvent event) {
        String message = event.getMessage();
        
        // "!ai " 로 시작하는 채팅만 감지
        if (message.startsWith("!ai ")) {
            event.setCancelled(true); // 원래 채팅은 숨김 처리
            String question = message.substring(4);
            String playerName = event.getPlayer().getName();

            // 플레이어에게 로딩 메시지 전송
            event.getPlayer().sendMessage(ChatColor.YELLOW + "[AI] " + ChatColor.GRAY + "생각 중입니다...");

            // 비동기로 파이썬 서버에 요청 보내기 (서버 렉 방지)
            Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
                String answer = askPythonServer(question);
                // 결과를 전체 채팅창에 브로드캐스트
                Bukkit.broadcastMessage(ChatColor.AQUA + playerName + "의 질문: " + ChatColor.WHITE + question);
                Bukkit.broadcastMessage(ChatColor.YELLOW + "[AI 답변] " + ChatColor.WHITE + answer);
            });
        }
    }

    // 파이썬 FastAPI 서버로 HTTP POST 요청을 보내는 함수
    private String askPythonServer(String question) {
        try {
            // JSON 데이터 생성
            String jsonInputString = "{\"question\": \"" + question.replace("\"", "\\\"") + "\"}";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://127.0.0.1:8000/ask")) // 파이썬 서버 주소
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonInputString))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            
            // 아주 간단한 JSON 파싱 (실제로는 Gson이나 Gson 라이브러리 사용 권장)
            String body = response.body();
            String answer = body.split("\"answer\":\"")[1].split("\"}")[0];
            
            // 줄바꿈 문자(\n)를 마인크래프트 채팅 줄바꿈으로 변환
            return answer.replace("\\n", "\n");

        } catch (Exception e) {
            e.printStackTrace();
            return "서버와 연결할 수 없습니다.";
        }
    }
}
```

---

## 🚀 4단계: 실행 및 테스트

1. **파이썬 API 서버 켜기:** 터미널에서 `api.py`가 있는 폴더로 이동 후 아래 명령어 실행
   ```bash
   ./venv/bin/python -m uvicorn api:app --reload
   ```
2. **플러그인 빌드 및 서버 적용:** 
   * IntelliJ에서 `Maven` 탭 -> `Lifecycle` -> `package`를 더블클릭하여 `.jar` 파일을 생성합니다.
   * 생성된 `.jar` 파일을 PaperMC 서버의 `plugins` 폴더에 넣고 마인크래프트 서버를 켭니다.
3. **인게임 테스트:**
   * 마인크래프트에 접속하여 채팅창에 `!ai 철 곡괭이 어떻게 만들어?` 라고 칩니다.
   * 파이썬 서버가 요청을 받아 처리한 뒤, 인게임 채팅창에 AI의 답변이 깔끔하게 올라오는 것을 확인할 수 있습니다!

## 💡 추가 팁 (인게임 최적화)
* 인게임 채팅창은 한 줄에 들어갈 수 있는 글자 수가 제한적입니다. 파이썬 쪽 프롬프트에서 **"최대 3문장 이내로 아주 짧게 요약해서 대답해"**라고 지시하는 것이 좋습니다.
* 마크다운(Markdown) 문법(`**`, `#` 등)은 인게임 채팅창에서 깨지므로, 파이썬 서버에서 답변을 보낼 때 마크다운 기호를 제거하고 보내는 전처리 과정이 필요합니다.