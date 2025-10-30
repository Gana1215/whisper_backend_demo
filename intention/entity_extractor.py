# Rule-based entity extraction functions
# ===============================================
# 🧠 Entity Extractor for Banking Intentions
# -----------------------------------------------
# Detects:
#   💰 Amounts (e.g. "100 мянга", "50,000", "2 сая ₮")
#   💳 Account numbers (8–12 digits)
#   💱 Currency names (₮, доллар, евро, юань)
#   📅 Date-like mentions (e.g. "өнөөдөр", "маргааш")
# ===============================================

import re
from typing import Dict, Any

# -----------------------------------------------
# ✅ Regex Patterns
# -----------------------------------------------
AMOUNT_PATTERN = re.compile(r"(\d{1,3}(?:[ ,.]?\d{3})*(?:\s*(?:мянга|сая))?)")
ACCOUNT_PATTERN = re.compile(r"\b\d{8,12}\b")
CURRENCY_PATTERN = re.compile(r"(₮|төгрөг|доллар|юань|евро)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"(өнөөдөр|маргааш|дараа\s*өдөр|өнгөрсөн\s*сар)", re.IGNORECASE)

# -----------------------------------------------
# ✅ Main Extraction Function
# -----------------------------------------------
def extract_entities(text: str) -> Dict[str, Any]:
    """Extract entities from Mongolian banking-related text."""
    entities: Dict[str, Any] = {}

    # --- Amount ---
    amounts = AMOUNT_PATTERN.findall(text)
    if amounts:
        entities["amounts"] = [a.strip() for a in amounts if a.strip()]

    # --- Account Numbers ---
    accounts = ACCOUNT_PATTERN.findall(text)
    if accounts:
        entities["accounts"] = accounts

    # --- Currency ---
    currencies = CURRENCY_PATTERN.findall(text)
    if currencies:
        entities["currencies"] = list(set(currencies))  # unique

    # --- Dates / temporal ---
    dates = DATE_PATTERN.findall(text)
    if dates:
        entities["dates"] = list(set(dates))

    return entities

# -----------------------------------------------
# ✅ Self-test (run standalone)
# -----------------------------------------------
if __name__ == "__main__":
    samples = [
        "50000 төгрөг шилжүүлээч",
        "Би 123456789012 данс руу 1 сая ₮ илгээмээр байна",
        "Юанийн ханш өнөөдөр хэд вэ?",
        "Өнөөдөр 50,000 доллар солимоор байна"
    ]
    for s in samples:
        print(f"\n🧩 Text: {s}")
        print("→", extract_entities(s))
