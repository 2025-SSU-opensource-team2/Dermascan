# main.py (수정)

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
import json
import urllib.request

# DB 관련 임포트 (User 모델 추가)
from sqlalchemy.orm import Session
from database import SessionLocal, init_db, Prediction, DiseaseInfo, User # User 모델 임포트 추가

# Pydantic 스키마 정의 (추가)
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel): # 회원가입 요청 시 사용
    email: EmailStr
    password: str

class UserResponse(BaseModel): # 사용자 정보 응답 시 사용
    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel): # 로그인 응답 시 사용
    access_token: str
    token_type: str

# 앱 생성
app = FastAPI()

# CORS 설정 (http://localhost:5173 추가)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://skin-diagnosis-fastapi-t6hz.vercel.app",
        "http://localhost:5173" # 요청하신 로컬 주소 추가
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 관련 설정
SECRET_KEY = os.getenv("SECRET_KEY", "your-fallback-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # 토큰 만료 시간 30분 -> 60분 (선택 사항)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 암호화 키 (환경 변수에서 가져오거나, 빌드 시 생성된 고정 키 사용)
ENCRYPTION_KEY = os.getenv("FERNET_KEY")
if ENCRYPTION_KEY:
    fernet = Fernet(ENCRYPTION_KEY.encode())
else:
    print("Warning: FERNET_KEY environment variable not set. Using a default key.")
    ENCRYPTION_KEY = b'a_default_safe_fernet_key_for_testing_purposes_for_fernet=' # 유효한 32바이트 URL-safe base64 인코딩된 키로 교체 필요
    fernet = Fernet(ENCRYPTION_KEY)


# 저장 디렉토리 생성
SAVE_DIR = "uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

# 클래스 라벨 (예시용)
class_labels = [f"Class_{i}" for i in range(31)]

# 모델 로딩
device = torch.device("cpu")

MODEL_URL = "https://storage.googleapis.com/dermascan-model-data-2025/best_model_epoch_19.pth"
LOCAL_MODEL_PATH = "best_model_epoch_19.pth"

try:
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
    print(f"Model downloaded to {LOCAL_MODEL_PATH}")

    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 31)
    model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {LOCAL_MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {LOCAL_MODEL_PATH}. This should not happen after download.")
    raise RuntimeError(f"Model file not found at {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    import sys
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    raise RuntimeError(f"Failed to load model: {e}")


# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 루트 엔드포인트 추가 (Health Check 응답용)
@app.get("/")
async def root():
    return {"message": "API is running successfully!"}

# JWT 생성 함수 (이하 동일)
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub") # 'sub'를 이메일로 사용
        if email is None:
            raise credentials_exception
        return email # 이메일 반환
    except JWTError:
        raise credentials_exception

# get_current_user 함수 수정: User 객체를 반환하도록
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = verify_token(token, credentials_exception)
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user # 사용자 객체 반환

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    init_db()
    from load_disease_data import load_data_to_db
    load_data_to_db()

# --- 사용자 관련 엔드포인트 추가 ---

@app.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_create: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_create.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    hashed_password = pwd_context.hash(user_create.password)
    new_user = User(email=user_create.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token) # 로그인 엔드포인트 수정
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


# --- 기존 코드 (user 파라미터 타입 변경: str -> User) ---
def validate_image_file(file: UploadFile):
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png files are allowed.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user), # User 객체로 변경
    db: Session = Depends(get_db)
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

        db_prediction = Prediction(
            user_id=current_user.email, # user.id 대신 user.email 사용 (고유 식별자로)
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

@app.get("/predictions/")
async def get_predictions(
    current_user: User = Depends(get_current_user), # User 객체로 변경
    db: Session = Depends(get_db)
):
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.email).all() # user_id를 email로 변경

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

@app.get("/diseases/{disease_name}")
async def get_disease_info(
    disease_name: str,
    db: Session = Depends(get_db)
):
    disease_info = db.query(DiseaseInfo).filter(DiseaseInfo.disease_name == disease_name).first()

    if not disease_info:
        raise HTTPException(status_code=404, detail="Disease information not found.")

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
        })
    return JSONResponse(content=results)