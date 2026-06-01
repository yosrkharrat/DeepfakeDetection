"""Celeb-DF v2 face-crop dataset loader."""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CelebDFDataset(Dataset):
    """
    Celeb-DF v2 frame loader.

    Expected directory structure:
        <root_dir>/
            real/   *.jpg / *.png  (YouTube-real or Celeb-real face crops)
            fake/   *.jpg / *.png  (Celeb-synthesis face crops)

    Labels: 0 = real, 1 = fake
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for label, folder in [(0, "real"), (1, "fake")]:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
