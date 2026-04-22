"""FFT transform utility.

Re-exports compute_fft_magnitude from src.models.fft_stream so the data
pipeline can import FFT logic without pulling in model classes.
"""

from src.models.fft_stream import compute_fft_magnitude  # noqa: F401
