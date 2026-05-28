from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

DATA_ROOT = Path('data/processed/processed/FaceForensics++_C23')
if not DATA_ROOT.exists():
    raise SystemExit(f'Dataset root not found: {DATA_ROOT}')

# Try to detect ImageFolder structure (expects class subfolders)
classes = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()][:5]
print('Top-level directories (sample):', classes)

# Build a simple transform using torchvision for quick test
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
print('Found classes:', dataset.classes)
print('Dataset length:', len(dataset))

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# Build model (EfficientNet-B4 backbone + small head)
weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
backbone = models.efficientnet_b4(weights=weights)
backbone_body = nn.Sequential(*list(backbone.children())[:-1])

in_features = backbone.classifier[1].in_features
model = nn.Sequential(
    backbone_body,
    nn.Flatten(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3*0.7),
    nn.Linear(256, len(dataset.classes)),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Run one batch
for images, labels in loader:
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
    print('Batch images:', images.shape)
    print('Logits:', logits.shape)
    break

print('RGB smoke-check passed.')
