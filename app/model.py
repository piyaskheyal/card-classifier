import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from PIL import Image
import timm
import torch.nn as nn
from io import BytesIO

# === Load class mapping from training directory ===
train_dataset = ImageFolder(root='card-classifier/data/train', transform=None)  # Adjust path
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# === Define model ===
class CardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# === Load model and weights ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardClassifier(num_classes=len(idx_to_class))
model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
model.to(device)
model.eval()

# === Define transform ===
transform = transforms.Compose([
    transforms.Resize(320, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Predict from image bytes ===
def predict_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = probs.max(1)

    return {
        "label": idx_to_class[pred.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
