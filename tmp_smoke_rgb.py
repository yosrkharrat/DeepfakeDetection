import torch
from torchvision import models
print('torch', torch.__version__)
try:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    m = models.efficientnet_b4(weights=weights)
    print('EfficientNet-B4 instantiated with weights')
except Exception as e:
    print('EfficientNet-B4 instantiation failed:', e)

x = torch.rand(2,3,224,224)
logits = m(x)
print('Output shape', logits.shape)
