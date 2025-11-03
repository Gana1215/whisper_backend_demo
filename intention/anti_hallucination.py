# ===============================================
# 🧠 Anti-Hallucination Defense v3.0
# -----------------------------------------------
# Removes meaningless repetition or broken syllables
# Keeps clean Mongolian words only
# ===============================================

import re

def defend(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # 🔹 Remove duplicated sequences like “энэ энэ энэ энэ…”
    t = re.sub(r"(?:\b(\w+)\b(?:\s+\1\b){2,})", r"\1", t)

    # 🔹 Remove repeated short junk tokens (“айз эй айз…”)
    t = re.sub(r"(\b[а-яА-Яa-zA-Z]{1,3}\b\s*){3,}", "", t)

    # 🔹 Collapse multiple spaces & trim
    t = re.sub(r"\s+", " ", t).strip()

    # 🔹 Reject if too short or non-Mongolian
    if len(t) < 4 or not re.search(r"[А-Яа-я]", t):
        return ""

    return t
