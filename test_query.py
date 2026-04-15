import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 1. 환경 변수 로드 (.env의 GOOGLE_API_KEY 인식)
load_dotenv()

# 2. 기존 로컬 벡터 DB 및 임베딩 모델 로드 (데이터 재수집 생략)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# persist_directory에 저장된 기존 DB를 불러옵니다.
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. 검색 (Retrieval)
query = "방패를 어떻게 만들어?"
retrieved_docs = vectorstore.similarity_search(query, k=3)

# 검색된 문서가 있으면 텍스트를 추출하고, 없으면 예외 처리용 텍스트를 할당합니다.
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "관련 정보를 찾을 수 없습니다."

# 4. LLM 설정 (Google Gemini)
# 답변 생성을 위해 gemini-1.5-flash 모델을 사용하며, 사실 기반 답변을 위해 창의성(temperature)을 0으로 설정합니다.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 5. 프롬프트 엔지니어링 (Prompt Engineering)
# LLM에게 역할과 제약 사항을 명확히 부여합니다.
prompt_template = """
당신은 마인크래프트 게임의 지능형 가이드 챗봇입니다.
아래 제공된 [Context]만을 사용하여 사용자의 [Question]에 답변하십시오.
Context에 없는 내용은 절대 추측해서 답변하지 마십시오.

조합법을 설명할 때 [Context]에 [1번칸:아이템명] 같은 3x3 제작 배치도 정보가 있다면,
반드시 '제작대 3x3 칸 기준'으로 위치를 명확히 설명해주세요.
(예: 1번칸(왼쪽 위)에는 참나무 판자, 2번칸(가운데 위)에는 철 주괴...)

[Context]
{context}

[Question]
{question}
"""

prompt = PromptTemplate.from_template(prompt_template)
final_prompt = prompt.format(context=context_text, question=query)

# 6. LLM 응답 요청 및 출력 (Generation)
print(f"사용자 질문: {query}\n")
print(f"--- 검색된 Context ---\n{context_text}\n----------------------")
print("답변 생성 중...")

response = llm.invoke(final_prompt)

print("\n--- 최종 LLM 응답 ---")
print(response.content)