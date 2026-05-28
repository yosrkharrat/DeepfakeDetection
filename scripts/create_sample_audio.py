import numpy as np
import soundfile as sf
from pathlib import Path
Path('data/processed').mkdir(parents=True, exist_ok=True)
fs = 16000
t = np.linspace(0, 3, int(3*fs), endpoint=False)
# synthetic speech-like signal: sum of two sinusoids
sig = 0.3*np.sin(2*np.pi*220*t) + 0.15*np.sin(2*np.pi*440*t)
# add small noise
sig += 0.02*np.random.randn(len(t))
sf.write('data/processed/sample_audio.wav', sig.astype('float32'), fs)
print('Wrote data/processed/sample_audio.wav')
