from faster_whisper import WhisperModel
import os

model = WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu")
print("Model type:", type(model))
print("Files:", os.listdir("models--gana1215--MN_Whisper_Small_CT2"))




python - <<'PY'
import os, asyncio, edge_tts
os.makedirs("static/tts", exist_ok=True)
path = "static/tts/diagnostic_test.wav"

async def go():
    tts = edge_tts.Communicate("Таны үлдэгдэл 1,250,000 төгрөг байна."
, voice="mn-MN-YesuiNeural")
    await tts.save(path)
    print("✅ Saved:", os.path.exists(path), path)

asyncio.run(go())
PY
