from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import io
import os
from cryptography.fernet import Fernet
import json # scores 저장을 위해 추가

# DB 관련 임포트
from sqlalchemy.orm import Session
from database import SessionLocal, init_db, Prediction, DiseaseInfo # DiseaseInfo 모델 임포트 추가

# 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skin-diagnosis-fastapi-t6hz.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 관련 설정
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 암호화 키 (보안 위해 .env로 옮기는 게 이상적)
# NOTE: 서버 재시작 시마다 키가 바뀌면 기존 암호화된 파일을 복호화할 수 없으므로,
# 실제 배포 시에는 환경 변수나 파일에서 고정된 키를 로드해야 합니다.
# 아래는 예시 키이며, 실제로는 generate_key.py로 생성한 유효한 키를 사용해야 합니다.
ENCRYPTION_KEY = b'S6JvrESVf8nRx0OkyZrjDiP7vKiFszecVrekF6MWgkM=' # 유효한 32바이트 URL-safe base64 인코딩된 키
fernet = Fernet(ENCRYPTION_KEY)

# 저장 디렉토리 생성
SAVE_DIR = "uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

# 클래스 라벨 (예시용)
class_labels = [f"Class_{i}" for i in range(31)]

# 모델 로딩
device = torch.device("cpu")
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 31)
# NOTE: best_model_epoch_19.pth 파일이 프로젝트 루트에 있다고 가정
model.load_state_dict(torch.load("best_model_epoch_19.pth", map_location=device))
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# JWT 생성 함수
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# JWT 검증 함수
def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise credentials_exception

# 현재 유저 가져오기
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    username = verify_token(token, credentials_exception)
    return username

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 애플리케이션 시작 시 DB 초기화
@app.on_event("startup")
async def startup_event():
    init_db()

# 로그인 엔드포인트
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != "admin" or form_data.password != "1234":
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 이미지 유효성 검증
def validate_image_file(file: UploadFile):
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png files are allowed.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

# 예측 엔드포인트
@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db) # DB 세션 주입
):
    try:
        validate_image_file(file)
        image_bytes = await file.read()

        timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
        filename = f"{timestamp}_{file.filename}"
        raw_path = os.path.join(SAVE_DIR, filename)
        encrypted_filename = f"enc_{filename}.bin"
        encrypted_path = os.path.join(SAVE_DIR, encrypted_filename)

        with open(raw_path, "wb") as f:
            f.write(image_bytes)

        with open(encrypted_path, "wb") as f:
            f.write(fernet.encrypt(image_bytes))

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]
            scores = {label: float(score) for label, score in zip(class_labels, outputs[0].tolist())}

        # DB에 예측 결과 저장
        db_prediction = Prediction(
            user_id=user,
            predicted_class=predicted_class,
            scores_json=json.dumps(scores),
            image_path_raw=raw_path,
            image_path_encrypted=encrypted_path
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "scores": scores,
            "saved_encrypted_path": encrypted_path,
            "prediction_id": db_prediction.id
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# 과거 예측 결과 조회 엔드포인트
@app.get("/predictions/")
async def get_predictions(
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    predictions = db.query(Prediction).filter(Prediction.user_id == user).all()

    results = []
    for pred in predictions:
        results.append({
            "id": pred.id,
            "user_id": pred.user_id,
            "predicted_class": pred.predicted_class,
            "scores": json.loads(pred.scores_json),
            "image_path_raw": pred.image_path_raw,
            "image_path_encrypted": pred.image_path_encrypted,
            "created_at": pred.created_at.isoformat()
        })
    return JSONResponse(content=results)

# 병명 정보 조회 엔드포인트 (새로 추가)
@app.get("/diseases/{disease_name}")
async def get_disease_info(
    disease_name: str,
    db: Session = Depends(get_db)
):
    disease_info = db.query(DiseaseInfo).filter(DiseaseInfo.disease_name == disease_name).first()

    if not disease_info:
        raise HTTPException(status_code=404, detail="Disease information not found.")

    # DB에서 가져온 JSON 문자열 필드를 다시 파이썬 객체로 변환
    symptoms = json.loads(disease_info.symptoms) if disease_info.symptoms else []
    causes = json.loads(disease_info.causes) if disease_info.causes else []
    treatment_methods = json.loads(disease_info.treatment_methods) if disease_info.treatment_methods else []
    recommended_medicine = json.loads(disease_info.recommended_medicine) if disease_info.recommended_medicine else []
    precautions = json.loads(disease_info.precautions) if disease_info.precautions else []

    return JSONResponse(content={
        "id": disease_info.id,
        "disease_name": disease_info.disease_name,
        "definition": disease_info.definition,
        "symptoms": symptoms,
        "causes": causes,
        "treatment_methods": treatment_methods,
        "recommended_medicine": recommended_medicine,
        "precautions": precautions,
        "source_url": disease_info.source_url,
        "last_updated": disease_info.last_updated.isoformat()
    })

# 모든 병명 정보 조회 엔드포인트 (선택 사항, 리스트 필요 시)
@app.get("/diseases/")
async def get_all_diseases_info(
    db: Session = Depends(get_db)
):
    all_diseases = db.query(DiseaseInfo).all()
    
    results = []
    for disease in all_diseases:
        results.append({
            "id": disease.id,
            "disease_name": disease.disease_name,
            "definition": disease.definition,
            "source_url": disease.source_url,
            "last_updated": disease.last_updated.isoformat()
            # 모든 필드를 반환할 수도 있지만, 목록 조회 시에는 필요한 정보만 반환하는 것이 효율적
        })
    return JSONResponse(content=results)