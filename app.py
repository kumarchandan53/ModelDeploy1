from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os

app = FastAPI()

# -------------------------------
# Load model
# -------------------------------
num_classes = 41  # Make sure this matches your checkpoint
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_best.pth", map_location="cpu"))
model.eval()  # Important: inference mode
# -------------------------------

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension

    # Inference with no_grad to save memory
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {"predicted_class": int(predicted.item())}

# -------------------------------
# Run with dynamic port for Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
