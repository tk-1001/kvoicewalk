import warnings

import numpy as np
import torch
from kokoro import KPipeline


class SpeechGenerator:
    def __init__(self):
        surpressWarnings()
        self.pipeline = KPipeline(lang_code="a",repo_id='hexgrad/Kokoro-82M')

        self._text_cache = {}

    def generate_audio(self, text: str, voice: torch.Tensor,speed: float = 1.0) -> np.typing.NDArray[np.float32]:
        processed = self._preprocess_text(text)

        voice_cpu = voice.cpu() if voice.is_cuda else voice
        generator = self.pipeline(processed, voice_cpu, speed)
        audio = []
        for gs, ps, chunk in generator:
            audio.append(chunk)
        return np.concatenate(audio)
    
    def _preprocess_text(self, text: str):
        if text not in self._text_cache:
            self._text_cache[text] = text
        return self._text_cache[text]

def surpressWarnings():
    # Surpress all these warnings showing up from libraries cluttering the console
    warnings.filterwarnings(
        "ignore",
        message=".*RNN module weights are not part of single contiguous chunk of memory.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore", message=".*is deprecated in favor of*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dropout option adds dropout after all but last recurrent layer*",
        category=UserWarning,
    )
