# Minecraft RAG AI 챗봇 개발 히스토리 (Part 3: 평가, 배포 및 UI 고도화)

## 1. RAG 성능 평가 기준 (The RAG Triad)
상용 서비스 수준의 RAG를 구축하고 평가하기 위한 3대 핵심 지표 학습 및 적용 방안 논의.
* **Context Relevance (문맥 적합성):** DB에서 사용자의 질문과 실제로 관련된 유의미한 문서(Chunk)를 잘 검색해 왔는가?
* **Faithfulness / Groundedness (사실 부합성):** AI의 답변이 오직 검색된 문서(Context)에만 근거하고 있는가? (환각 현상 방지)
* **Answer Relevance (답변 관련성):** 최종 답변이 사용자의 원래 질문 의도에 정확히 부합하는가?

## 2. RAG 평가 방법론
* **출처 표기(Citation):** 답변 시 참고한 문서의 출처를 명시하여 검증 가능성 확보.
* **네거티브 테스트(Negative Testing):** DB에 없는 내용을 질문하여 AI가 "모른다"고 답변하는지(환각 방지) 테스트. (현재 프롬프트에 적용 완료)
* **LLM-as-a-Judge:** RAGAS, TruLens 등의 프레임워크를 활용하여 LLM이 직접 RAG의 3대 지표를 자동 채점하는 최신 트렌드 학습.

## 3. 포트폴리오용 웹 배포(Deployment) 전략
* 기존 Vercel 블로그 연동의 기술적 한계(Python/Streamlit 및 로컬 DB 호스팅 불가) 확인.
* **Streamlit Community Cloud**를 최적의 배포 플랫폼으로 선정.
* 배포를 위한 `requirements.txt` 생성 및 GitHub 연동, 환경 변수(Secrets) 설정 방법 정리.

## 4. UI/UX 및 기능 고도화 (app.py)
* **디자인 개선:** 기존 텍스트 이모티콘(⛏️)을 실제 마인크래프트 로고 이미지(`마크로고.webp`)로 교체하여 시각적 완성도 향상.
* **레이아웃 최적화:** `st.columns`를 활용하여 로고와 타이틀을 깔끔하게 병렬 배치.
* **편의 기능 추가:** 사이드바(Sidebar)를 도입하여 '대화 초기화' 버튼 및 챗봇 정보(데이터 소스, AI 모델) 패널 추가.
