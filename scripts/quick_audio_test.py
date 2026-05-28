import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
from src.api.utils import audio_utils
from src.models.audio_stream import MFCCCNN, Wav2Vec2AudioClassifier
from src.utils.metrics import probabilities_from_logits
import torch

AUDIO_FILE = os.environ.get('AUDIO_FILE') or 'data/processed/sample_audio.wav'
MODEL_TYPE = os.environ.get('AUDIO_MODEL_TYPE') or 'mfcc_cnn'  # or 'wav2vec'
MODEL_ID = os.environ.get('AUDIO_MODEL_ID') or 'facebook/wav2vec2-base'

if not Path(AUDIO_FILE).exists():
    print('Audio file not found:', AUDIO_FILE)
    sys.exit(1)

waveform, sr = audio_utils.load_waveform(AUDIO_FILE, sr=16000)

if MODEL_TYPE == 'wav2vec':
    try:
        model = Wav2Vec2AudioClassifier.from_pretrained(MODEL_ID)
    except Exception as e:
        print('Failed to load Wav2Vec2:', e)
        sys.exit(1)
    logits = model.predict_logits(waveform, sr=sr)
else:
    model = MFCCCNN()
    # If a checkpoint path is provided use it
    ck = os.environ.get('AUDIO_CHECKPOINT')
    if ck and Path(ck).exists():
        sd = torch.load(ck, map_location='cpu')
        try:
            model.load_state_dict(sd)
        except Exception:
            # try nested dict formats
            if isinstance(sd, dict):
                for k in ('state_dict', 'model_state_dict', 'model'):
                    if k in sd and isinstance(sd[k], dict):
                        model.load_state_dict(sd[k])
                        break
    logits = model.predict_logits(waveform, sr=sr)

probs = probabilities_from_logits(logits).numpy()
print('Probs:', probs)
