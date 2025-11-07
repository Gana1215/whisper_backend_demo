# ===============================================================
# 🧠 train_intent_model.py (v5.1 — Context-Aware Edition)
# ---------------------------------------------------------------
# ✅ Extends TF-IDF word context to 1–4 n-grams for deeper intent capture
# ✅ Keeps char(2–5) for fuzzy Mongolian spellings
# ✅ Retains semantic centroids + typo correction + calibration
# ✅ 100 % compatible with FastAPI dynamic router
# ===============================================================

import os, sys, re, joblib, random, difflib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1️⃣ Paths & seed ---
ROOT = Path(__file__).resolve().parents[1]
INTENT_DIR = ROOT / "intention"
CSV_PATH = INTENT_DIR / "intents.csv"
VEC_PATH = INTENT_DIR / "tfidf_vectorizer.pkl"
CLF_PATH = INTENT_DIR / "intent_classifier.pkl"
CENT_PATH = INTENT_DIR / "intent_centroids.pkl"  # semantic centroids

SEED = 42
random.seed(SEED)

# --- 2️⃣ Load dataset ---
if not CSV_PATH.exists():
    print(f"❌ Missing {CSV_PATH}")
    sys.exit(1)
df = pd.read_csv(CSV_PATH).dropna()
if not {"text", "intent"}.issubset(df.columns):
    print("❌ intents.csv must have columns: text,intent")
    sys.exit(1)

df["intent"] = df["intent"].astype(str).str.strip().str.lower()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
unique_intents = sorted(df["intent"].unique())
print(f"📄 Dataset size: {len(df)} | Intents: {len(unique_intents)}")

# --- 3️⃣ Fuzzy auto-correct (data-driven vocabulary) ---
vocab = Counter()
for text in df["text"]:
    for w in re.findall(r"[а-яa-z0-9өү]+", text.lower()):
        vocab[w] += 1
COMMON_WORDS = [w for w, c in vocab.items() if c > 1]

def fuzzy_fix(text: str) -> str:
    words = text.split()
    fixed = []
    for w in words:
        m = difflib.get_close_matches(w, COMMON_WORDS, n=1, cutoff=0.7)
        fixed.append(m[0] if m else w)
    return " ".join(fixed)

def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^а-яa-z0-9өү\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = fuzzy_fix(t)
    return t.strip()

df["text_norm"] = df["text"].astype(str).apply(normalize_text)

# --- 4️⃣ Train/test split ---
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"], df["intent"], test_size=0.2,
        random_state=SEED, stratify=df["intent"]
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"], df["intent"], test_size=0.2, random_state=SEED
    )

# --- 5️⃣ TF-IDF Vectorizers ---
# Character-level handles typos / morphological variations
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=10000
)

# Word-level extended to 4-grams for richer context
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 4),
    max_features=12000
)

# Combine both feature spaces
vectorizer = FeatureUnion([
    ("char", char_vec),
    ("word", word_vec)
])

Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

# --- 6️⃣ Classifier ---
base = LogisticRegression(
    C=2.0,
    solver="lbfgs",
    max_iter=4000,
    class_weight="balanced"
)
clf = CalibratedClassifierCV(base, cv=3)
clf.fit(Xtr, y_train)

# --- 7️⃣ Build semantic centroids ---
intent_centroids = {}
for intent in unique_intents:
    idx = df["intent"] == intent
    if idx.sum() > 0:
        vecs = vectorizer.transform(df.loc[idx, "text_norm"])
        intent_centroids[intent] = np.asarray(vecs.mean(axis=0))

# --- 8️⃣ Evaluate ---
pred = clf.predict(Xte)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")

print("\n✅ Evaluation Results")
print(f"🎯 Accuracy: {acc:.4f} | 📊 Macro F1: {f1:.4f}\n")
print(classification_report(y_test, pred, digits=3))

# --- 9️⃣ Save artifacts ---
INTENT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(vectorizer, VEC_PATH)
joblib.dump(clf, CLF_PATH)
joblib.dump(intent_centroids, CENT_PATH)

print(f"\n💾 Saved: {VEC_PATH.name}, {CLF_PATH.name}, {CENT_PATH.name}")
print("🚀 Model ready for FastAPI dynamic intent router.")
