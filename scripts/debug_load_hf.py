import traceback
from src.models.audio_stream import Wav2Vec2AudioClassifier
MODEL_ID = 'mo-thecreator/deepfake-audio-detection'
try:
    print('Loading', MODEL_ID)
    m = Wav2Vec2AudioClassifier.from_pretrained(MODEL_ID)
    print('Loaded OK:', type(m))
except Exception as e:
    traceback.print_exc()
    print('Error:', e)
