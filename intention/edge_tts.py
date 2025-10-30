# ===============================================
# 🎙️ Edge-TTS Synthesis Helper (Phase 2 — Final + DIAG)
# -----------------------------------------------
# ✅ Microsoft Edge-TTS voices (mn-MN-YesuiNeural / mn-MN-BataaNeural)
# ✅ Creates static/tts/ automatically
# ✅ Thread-safe async run
# ✅ Optional DIAG logging (env or param)
# ✅ Returns relative path (used by intent_router)
# ===============================================

import os
import asyncio
import edge_tts
from datetime import datetime

# =========================================================
# 🔹 Global DIAG toggle
# =========================================================
ENV_DIAG = os.getenv("DIAG", "0") == "1"

def dlog(*args, enabled=False):
    """Conditional debug logger"""
    if enabled:
        print(*args)

# =========================================================
# 🔹 Core synthesis
# =========================================================
def synthesize_tts(
    text: str,
    voice: str = "mn-MN-YesuiNeural",
    output_dir: str = "static/tts",
    basename: str = "tts_reply",
    DIAG: bool = ENV_DIAG,
) -> str:
    """
    Synthesize Mongolian speech using Edge-TTS.

    Args:
        text (str): text to synthesize
        voice (str): Edge-TTS voice name
        output_dir (str): output folder (e.g. static/tts)
        basename (str): base filename
        DIAG (bool): enable diagnostic logging

    Returns:
        str: relative path (e.g. 'static/tts/check_balance_reply_1730218895.wav')
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = int(datetime.now().timestamp())
    fname = f"{basename}_{ts}.wav"
    output_path = os.path.join(output_dir, fname)

    async def _run():
        try:
            dlog(f"🎧 [Edge-TTS] Synth → {voice}: {text}", enabled=DIAG)
            tts = edge_tts.Communicate(text, voice=voice)
            await tts.save(output_path)
            dlog(f"✅ [Edge-TTS] Saved → {output_path}", enabled=DIAG)
        except Exception as e:
            dlog(f"❌ [Edge-TTS] Failed: {e}", enabled=DIAG)

    # Safe run
    try:
        asyncio.run(_run())
    except RuntimeError:
        # If inside FastAPI event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_run())
        else:
            loop.run_until_complete(_run())

    if DIAG:
        exists = os.path.exists(output_path)
        print(f"📦 [Edge-TTS] File check → {exists}: {output_path}")

    return f"{output_dir}/{fname}"
