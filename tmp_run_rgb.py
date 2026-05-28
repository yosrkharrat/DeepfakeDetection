import os, sys
from pathlib import Path
REPO_ROOT = Path(r"C:\Users\yosrk\DeepFake")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, models
import albumentations as A
from albumentations.pytorch import ToTensorV2

print('torch', torch.__version__)

CFG = {
    'data_root': str(REPO_ROOT / 'data' / 'processed' / 'FaceForensics++_C23'),
    'img_size': 224,
    'batch_size': 32,
    'epochs': 1,
    'lr': 1e-4,
    'pretrained': True,
}

# transforms
_train_aug = A.Compose([
    A.Resize(CFG['img_size'], CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

# model
weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if CFG['pretrained'] else None
backbone = models.efficientnet_b4(weights=weights)
print('Backbone loaded')
# remove head
backbone_body = nn.Sequential(*list(backbone.children())[:-1])
# in_features
in_features = backbone.classifier[1].in_features
print('in_features', in_features)

model = nn.Sequential(
    backbone_body,
    nn.Flatten(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3*0.7),
    nn.Linear(256, 2),
)

print('Model constructed')
print('Total params', sum(p.numel() for p in model.parameters()))
print('Trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

# check data root
if not Path(CFG['data_root']).exists():
    print('Data root not found, skipping dataset creation:', CFG['data_root'])
else:
    ds = datasets.ImageFolder(CFG['data_root'])
    print('Found dataset with length', len(ds))

print('Done')
