import requests
from pathlib import Path
p = Path('data/processed/processed/FaceForensics++_C23/real/000/frame_000000_face_00.jpg')
url = 'http://127.0.0.1:5000/api/detect'
with p.open('rb') as f:
    files = {'file': ('frame.jpg', f, 'image/jpeg')}
    r = requests.post(url, files=files, timeout=10)
    print('STATUS', r.status_code)
    print(r.json())
