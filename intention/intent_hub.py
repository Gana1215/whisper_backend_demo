# intention/intent_hub.py
# ===============================================
# 🧠 Intent Hub (LOCKED)
# -----------------------------------------------
# Classification ONLY:
#   - vectorize
#   - classifier.predict
#   - confidence
#   - entities extraction
# Returns: intent, confidence, entities, base_intent/base_confidence, fallback_used
# ===============================================

import os
import csv
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from intention.entity_extractor import extract_entities


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENT_DIR = os.path.join(BASE_DIR, "intention")

VEC_PATH = os.path.join(INTENT_DIR, "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(INTENT_DIR, "intent_classifier.pkl")
CSV_PATH = os.path.join(INTENT_DIR, "intents.csv")

CONF_THRESH = float(os.getenv("INTENT_CONF_THRESHOLD", "0.25"))
FALLBACK_K = int(os.getenv("INTENT_FALLBACK_K", "3"))

vectorizer = joblib.load(VEC_PATH)
classifier = joblib.load(CLF_PATH)

# load training corpus for cosine fallback
_train_texts, _train_labels = [], []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    header = f.readline()
    if "text" not in header or "intent" not in header:
        f.seek(0)
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            _train_texts.append(row[0].strip())
            _train_labels.append(row[1].strip().lower())

_TRAIN_X = vectorizer.transform(_train_texts)
_TRAIN_Y = np.array(_train_labels)


def _majority_vote_topk(sims: np.ndarray, k: int) -> Tuple[str, float]:
    k = max(1, min(k, sims.shape[0]))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    top_labels = _TRAIN_Y[top_idx]
    counts, best_sim = {}, {}

    for i, lab in zip(top_idx, top_labels):
        counts[lab] = counts.get(lab, 0) + 1
        s = float(sims[i])
        if lab not in best_sim or s > best_sim[lab]:
            best_sim[lab] = s

    label = max(counts, key=lambda l: (counts[l], best_sim[l]))
    return label, float(best_sim[label])


def classify(text: str, dlog=None) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "intent": "error",
            "confidence": 0.0,
            "entities": {},
            "base_intent": "error",
            "base_confidence": 0.0,
            "fallback_used": False,
        }

    X = vectorizer.transform([text])
    pred = classifier.predict(X)[0]
    prob = float(classifier.predict_proba(X)[0].max())
    entities = extract_entities(text)

    if prob < CONF_THRESH:
        sims = cosine_similarity(X, _TRAIN_X)[0]
        voted, _best = _majority_vote_topk(sims, FALLBACK_K)
        dlog and dlog(f"🧭 Fallback top-{FALLBACK_K} → {voted} (base={pred}, prob={prob:.3f})")
        return {
            "intent": voted,
            "confidence": prob,
            "entities": entities,
            "base_intent": pred,
            "base_confidence": prob,
            "fallback_used": True,
        }

    return {
        "intent": pred,
        "confidence": prob,
        "entities": entities,
        "base_intent": pred,
        "base_confidence": prob,
        "fallback_used": False,
    }
#