# intention/mn_number_parser.py
# ===============================================
# 🇲🇳 Mongolian Number Word Parser (Golden, lightweight)
# -----------------------------------------------
# Goals:
#  - "Нэг сая таван зуун гучин долоон мянга гурван зуун наян ес" -> 1537889
#  - Handles common STT typos: "зун"->"зуун", "дөрөг"->"дөрөв", etc.
#  - Also supports digit-by-digit speech: "нэг хоёр гурав дөрөв" -> "1234"
#
# No heavy ML. Uses:
#  - normalization + alias map
#  - fuzzy correction (difflib) for near-miss tokens
# ===============================================

from __future__ import annotations

import re
import difflib
from typing import List, Optional, Tuple

# -----------------------------
# Canonical token dictionaries
# -----------------------------

ONES = {
    "тэг": 0,
    "нэг": 1,
    "хоёр": 2,
    "гурван": 3,
    "гурв": 3,
    "дөрвөн": 4,
    "дөрөв": 4,
    "тав": 5,
    "зургаа": 6,
    "долоо": 7,
    "найм": 8,
    "ес": 9,
    "есөн": 9,
}

TENS = {
    "арав": 10,
    "хорь": 20,
    "гуч": 30,
    "дөч": 40,
    "тавь": 50,
    "жар": 60,
    "дал": 70,
    "ная": 80,
    "ер": 90,
}

SCALES = {
    "зуун": 100,
    "зуу": 100,      # sometimes STT drops ending
    "мянга": 1000,
    "сая": 1_000_000,
    "тэрбум": 1_000_000_000,
}

# For money and noise tokens we want to ignore
NOISE = {
    "төгрөг",
    "төгрөгөөр",
    "төгрөгийг",
    "мөнгө",
    "₮",
    "төг",
    "төгр",
    "mn",
    "мөнгөө",
}

# -----------------------------
# Common STT confusions / aliases
# -----------------------------
ALIASES = {
    # zуун confusion
    "зун": "зуун",
    "зуунг": "зуун",
    "зууг": "зуун",

    # дөрөв confusion
    "дөрөг": "дөрөв",
    "дөрөгөө": "дөрөв",
    "дөрг": "дөрөв",

    # nine confusion
    "есө": "ес",
    "есөнг": "есөн",

    # tens partials
    "хори": "хорь",
    "гучи": "гуч",
    "дочи": "дөч",
    "тави": "тавь",
    "жари": "жар",
    "дали": "дал",
    "наяи": "ная",
    "ери": "ер",

    # scale suffix variants
    "мянган": "мянга",
    "саяын": "сая",
    "саяа": "сая",
    "тэрбумын": "тэрбум",
    "зуугийн": "зуун",

    # ones variants (STT sometimes adds vowels)
    "нэгэн": "нэг",
    "хоёрон": "хоёр",
    "гурва": "гурван",
    "дөрвө": "дөрөв",
    "тава": "тав",
}

# We will fuzzy-correct against this vocabulary set
VOCAB = set(ONES) | set(TENS) | set(SCALES) | NOISE

# -----------------------------
# Tokenization / normalization
# -----------------------------

def _clean_text(s: str) -> str:
    s = (s or "").lower().strip()
    # keep mn letters + digits + spaces
    s = re.sub(r"[^0-9а-яөүё\s\-]+", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _split_tokens(s: str) -> List[str]:
    s = _clean_text(s)
    if not s:
        return []
    return s.split()

def _alias_token(tok: str) -> str:
    return ALIASES.get(tok, tok)

def _fuzzy_fix(tok: str) -> str:
    """
    Lightweight fuzzy correction (edit-distance based).
    If token is close to a known vocab entry, map it.
    """
    if not tok or tok.isdigit():
        return tok
    if tok in VOCAB:
        return tok
    # apply aliases first
    tok2 = _alias_token(tok)
    if tok2 in VOCAB:
        return tok2

    # fuzzy match
    # cutoff is conservative to avoid wrong corrections
    matches = difflib.get_close_matches(tok2, list(VOCAB), n=1, cutoff=0.82)
    if matches:
        return matches[0]
    return tok2

def normalize_tokens(text: str) -> List[str]:
    toks = _split_tokens(text)
    out = []
    for t in toks:
        t = _fuzzy_fix(t)
        if not t:
            continue
        if t in NOISE:
            continue
        out.append(t)
    return out

# -----------------------------
# 1) Digit-by-digit speech
# -----------------------------

DIGIT_WORDS = {
    "тэг": "0",
    "нэг": "1",
    "хоёр": "2",
    "гурван": "3",
    "гурв": "3",
    "дөрөв": "4",
    "дөрвөн": "4",
    "тав": "5",
    "зургаа": "6",
    "долоо": "7",
    "найм": "8",
    "ес": "9",
    "есөн": "9",
}

def mn_words_to_digits(text: str) -> str:
    """
    Convert sequences like:
      "нэг хоёр гурав дөрөв" -> "1234"
    Also supports mixed digits:
      "нэг 2 гурван" -> "123"
    """
    toks = normalize_tokens(text)
    digits = []
    for t in toks:
        if t.isdigit():
            digits.append(t)
            continue
        if t in DIGIT_WORDS:
            digits.append(DIGIT_WORDS[t])
            continue
    return "".join(digits)

# -----------------------------
# 2) Number phrase -> integer
# -----------------------------

def _parse_under_100(tokens: List[str], i: int) -> Tuple[int, int]:
    """
    Parse tens/ones at tokens[i:].
    Returns (value, consumed)
    Examples:
      ["наян","ес"] -> 89
      ["арав"] -> 10
      ["гучин","долоо"] might appear as "гуч" "долоо" (we handle both)
    """
    if i >= len(tokens):
        return 0, 0

    t = tokens[i]

    # direct digit token
    if t.isdigit():
        return int(t), 1

    # tens
    if t in TENS:
        base = TENS[t]
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.isdigit():
                return base + int(nxt), 2
            if nxt in ONES:
                return base + ONES[nxt], 2
        return base, 1

    # ones
    if t in ONES:
        return ONES[t], 1

    return 0, 0

def mn_words_to_amount(text: str) -> Optional[int]:
    """
    Main parser for Mongolian number words.
    Strategy:
      - tokens normalized + fuzzy corrected
      - parse by segments separated by big scales (billion/million/thousand)
      - within each segment, parse hundreds + under-100

    Handles:
      "нэг сая таван зуун гучин долоон мянга гурван зуун наян ес" -> 1537889
    """
    # If digits already exist, prefer them (fast-path)
    digits = re.sub(r"\D+", "", text or "")
    if digits:
        try:
            v = int(digits)
            return v if v > 0 else None
        except:
            pass

    tokens = normalize_tokens(text)
    if not tokens:
        return None

    total = 0
    current = 0  # current chunk below 1000

    i = 0
    while i < len(tokens):
        t = tokens[i]

        # Big scales
        if t in ("тэрбум", "сая", "мянга"):
            scale = SCALES[t]
            # If "current" is 0 but token is like "сая" alone, treat as 1*scale
            if current == 0:
                current = 1
            total += current * scale
            current = 0
            i += 1
            continue

        # Hundred
        if t in ("зуун", "зуу"):
            # if "зуун" appears without leading number, assume 1 hundred
            if current == 0:
                current = 100
            else:
                # if current already has e.g. 5, then 5*100 = 500
                # but if current is like 23, this is ambiguous; we treat last under-100 as multiplier
                # safer approach: if current < 10 -> multiplier, else assume already built and add 100
                if current < 10:
                    current = current * 100
                else:
                    current += 100
            i += 1
            continue

        # Try parse under-100
        v, consumed = _parse_under_100(tokens, i)
        if consumed > 0:
            current += v
            i += consumed
            continue

        # Unknown token -> skip
        i += 1

    total += current

    if total <= 0:
        return None
    return total

def format_money_2dp(amount_value: Optional[int]) -> str:
    if not isinstance(amount_value, int):
        return ""
    return f"{amount_value:,.2f}"
