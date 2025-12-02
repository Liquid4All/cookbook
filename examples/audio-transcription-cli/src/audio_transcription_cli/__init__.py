"""Real-time audio-to-speech recognition using LFM2-Audio-1.5B."""

from .audio_preprocessing import save_raw_audio_as_wav
from .config import Config
from .model_wrapper import LFM2AudioWrapper

__version__ = "0.1.0"
__all__ = [
    "save_raw_audio_as_wav",
    "LFM2AudioWrapper",
    "Config",
]


def hello() -> str:
    return "Hello from audio-demo!"
