# 피부 진단 FastAPI 서버

## 🖼️ 기능 요약
- `/upload/`: 이미지 파일을 POST로 업로드
- `/files/{filename}`: 업로드된 이미지 확인용 GET 요청

## 🚀 실행 방법 (로컬)

1. 저장소 클론
```bash
git clone https://github.com/2025-SSU-opensource-team2/skin-diagnosis-fastapi.git
cd skin-diagnosis-fastapi
```

2. 가상 환경 설정 (선택)
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
```

3. 패키지 설치
```bash
pip install -r requirements.txt
```

4. 서버 실행
```bash
uvicorn main:app --reload
```

- 서버 주소: http://127.0.0.1:8000
- Swagger 문서: http://127.0.0.1:8000/docs
