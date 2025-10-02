from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import shutil

app = FastAPI()

# Load model
num_classes = 41
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_best.pth", map_location="cpu"))

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return int(predicted.item())

@app.get("/")
def home():
    return {"message": "Cattle Breed Prediction API is Running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = predict_image("temp.jpg")
    return {"prediction": result}
