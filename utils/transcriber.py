import ctranslate2
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor

MODEL_PATH = "models/MN_Whisper_Small_CT2"

print("🔁 Loading CTranslate2 Whisper model...")
translator = ctranslate2.Translator(MODEL_PATH, device="cpu")  # "cuda" if GPU later
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def transcribe_audio(file_bytes: bytes) -> str:
    """Transcribes WAV bytes using Whisper + CTranslate2."""
    with open("temp.wav", "wb") as f:
        f.write(file_bytes)
    audio, sr = sf.read("temp.wav")
    features = processor(audio, sampling_rate=sr, return_tensors="np").input_features
    results = translator.translate_batch(features.tolist())
    tokens = np.array(results[0].hypotheses[0])
    text = processor.decode(tokens)
    return text.strip()
