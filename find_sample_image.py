from pathlib import Path
root=Path('data/processed/processed/FaceForensics++_C23')
img=None
for p in root.rglob('*'):
    if p.suffix.lower() in {'.jpg','.jpeg','.png'}:
        img=p
        break
print(img)
