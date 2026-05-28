import sys
sys.path.insert(0, '.')
from src.api.utils import audio_utils
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch

MODEL_ID = 'mo-thecreator/deepfake-audio-detection'
waveform, sr = audio_utils.load_waveform('data/processed/sample_audio.wav', sr=16000)
print('Loaded waveform', waveform.shape, 'sr', sr)
print('Loading model', MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
print('Processor type:', type(processor))
inputs = processor(waveform, sampling_rate=sr, return_tensors='pt', padding=True)
print('Input keys:', inputs.keys())
with torch.no_grad():
    out = model(**{k: v.to(model.device) for k, v in inputs.items()})
    print('Logits:', out.logits)
    import numpy as np
    probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()
    print('Probs:', probs)
