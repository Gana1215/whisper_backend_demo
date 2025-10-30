# diagnose_phase2.py
import time, asyncio, edge_tts, joblib
from pydub import AudioSegment
from faster_whisper import WhisperModel

AUDIO_FILE = "test.wav"  # 🟢 put any short WAV (2–3s) here

t0 = time.perf_counter()
AudioSegment.from_file(AUDIO_FILE)
t1 = time.perf_counter()

model = WhisperModel("models/MN_Whisper_Small_CT2", device="cpu", compute_type="int8")
model.transcribe(AUDIO_FILE)
t2 = time.perf_counter()

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
joblib.load(os.path.join(BASE_DIR, "intention", "intent_model.joblib"))

t3 = time.perf_counter()

asyncio.run(edge_tts.Communicate("Харилцагч таны үлдэгдэл:", voice="mn-MN-YesuiNeural").save("out.mp3"))
t4 = time.perf_counter()

print(f"""
Decode: {(t1-t0)*1000:.1f} ms
ASR: {(t2-t1)*1000:.1f} ms
Intent: {(t3-t2)*1000:.1f} ms
TTS: {(t4-t3)*1000:.1f} ms
TOTAL: {(t4-t0)*1000:.1f} ms
""")
