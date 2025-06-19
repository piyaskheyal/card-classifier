import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn
import sys
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(root='card-classifier/data/train', transform=None)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# ===== Load the model =====
class CardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifier, self).__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Load model + weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardClassifier(num_classes=53)
model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
model.to(device)
model.eval()

# ===== Define transforms (same as during training) =====
from torchvision.transforms import InterpolationMode
transform = transforms.Compose([
    transforms.Resize(320, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== Load and preprocess image =====
img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Shape: (1, C, H, W)
input_tensor = input_tensor.to(device)

# ===== Inference =====
with torch.no_grad():
    output = model(input_tensor)  # Shape: (1, num_classes)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_class = probabilities.max(1)

# ===== Output =====
predicted_label = idx_to_class[predicted_class.item()]
print(f"Predicted class: {predicted_label}, Confidence: {confidence.item()*100:.2f}%")
