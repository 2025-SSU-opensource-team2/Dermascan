from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

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
async def predict_image(file: UploadFile = File(...)):
    try:
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
