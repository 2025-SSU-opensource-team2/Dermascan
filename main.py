from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
import shutil
import io
import torch
import torchvision.transforms as transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = torch.load("model.pt", map_location=torch.device("cpu"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),         
    transforms.Normalize([0.5]*3, [0.5]*3)  
])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/files/{filename}")
def get_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(file_path)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)  
   
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

  
    return JSONResponse(content={"result": pred})
