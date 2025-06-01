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
ENCRYPTION_KEY = Fernet.generate_key()
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
    user: str = Depends(get_current_user)
):
    try:
        validate_image_file(file)
        image_bytes = await file.read()

        # 🛠 파일 이름에서 ":" 제거 (윈도우 저장 오류 해결)
        timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
        filename = f"{timestamp}_{file.filename}"
        raw_path = os.path.join(SAVE_DIR, filename)
        encrypted_path = os.path.join(SAVE_DIR, f"enc_{filename}.bin")

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

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "scores": scores,
            "saved_encrypted_path": encrypted_path
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")