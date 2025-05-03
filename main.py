from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# 앱 생성
app = FastAPI()

# CORS 설정 (프론트와 연동 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 시에는 허용 도메인 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 관련 설정
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise credentials_exception

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != "admin" or form_data.password != "1234":
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    username = verify_token(token, credentials_exception)
    return username

# 이미지 파일 검증 함수
def validate_image_file(file: UploadFile):
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png files are allowed.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

# 클래스 라벨 (예시 - 0번부터 30번까지 31개)
class_labels = [f"Class_{i}" for i in range(31)]

# 모델 로딩
device = torch.device("cpu")
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 31)  # ← 31개 클래스
model.load_state_dict(torch.load("best_model_epoch_19.pth", map_location=device))
model.to(device)
model.eval()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 예측 API
@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user)
):
    try:
        validate_image_file(file)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]
            scores = {label: float(score) for label, score in zip(class_labels, outputs[0].tolist())}

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "scores": scores
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
