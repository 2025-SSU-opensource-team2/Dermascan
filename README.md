# 모델 학습 코드
## 파일 설명
- `train.py` : 모델 학습 코드. 이 파일을 통해 학습한 pretrain.pth 파일을 전달할 예정. -> `sh train.sh` 명령으로 모델학습 가능. 
- `sample_index.css` : 선행 오픈소스의 css
- `public/sample_index.html` : 선행 오픈소스의 html
- `sample_main.py` : 선행 오픈소스의 main 파일. 이거 실행하면 웹에서 이미지 업로드해서 pth 파일 불러온 후 예측 병변 클래스 출력. (flask기반)
# 가상환경
- conda 환경으로 requirement.txt 설치 권장. 

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