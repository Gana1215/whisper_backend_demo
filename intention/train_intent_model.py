# ===============================================================
# 🧠 train_intent_model.py (v4.0 — Hybrid Confidence Edition)
# ---------------------------------------------------------------
# ✅ Dual TF-IDF: char(3–5) + word(1–3) for Mongolian language
# ✅ Normalizes “тусламж” vs “инфо” domains robustly
# ✅ Calibrated LogisticRegression for realistic probabilities
# ✅ Balanced class weights + detailed evaluation logs
# ✅ 100% compatible with FastAPI intent router loader
# ===============================================================

import os, sys, joblib, random, re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --- 1️⃣ Paths & seed ---
ROOT = Path(__file__).resolve().parents[1]
INTENT_DIR = ROOT / "intention"
CSV_PATH = INTENT_DIR / "intents.csv"
VEC_PATH = INTENT_DIR / "tfidf_vectorizer.pkl"
CLF_PATH = INTENT_DIR / "intent_classifier.pkl"

SEED = 42
random.seed(SEED)

# --- 2️⃣ Load & validate CSV ---
if not CSV_PATH.exists():
    print(f"❌ Missing {CSV_PATH}. Please place your final intents.csv there.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH).dropna()

if not {"text", "intent"}.issubset(df.columns):
    print("❌ intents.csv must have columns: text,intent")
    sys.exit(1)

df["intent"] = df["intent"].astype(str).str.strip().str.lower()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
unique_intents = sorted(df["intent"].unique().tolist())
print(f"📄 Dataset size: {len(df)} | Intents: {len(unique_intents)}")
print(f"🧩 Intents: {unique_intents}")

# --- 3️⃣ Smart Mongolian normalization ---
def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^а-яa-z0-9өү\s]", " ", t)
    t = re.sub(r"\s+", " ", t)

    # Support / Contact synonyms
    for p in [
        "харилцагчийн үйлчилгээ",
        "үйлчилгээний төв",
        "оператор",
        "дугаар",
        "утас",
        "холбож",
        "холбогдох",
        "холбоо барих",
    ]:
        t = t.replace(p, " тусламж ")

    # Info / Customer info synonyms
    for p in ["мэдээлэл", "шинэчлэх", "шалгах", "бүртгэл"]:
        t = t.replace(p, " инфо ")

    # Misc cleanup
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text_norm"] = df["text"].astype(str).apply(normalize_text)

# --- 4️⃣ Split ---
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"],
        df["intent"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["intent"],
    )
except ValueError:
    print("⚠️ Stratify failed — random split used.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"], df["intent"], test_size=0.2, random_state=SEED
    )

# --- 5️⃣ Hybrid TF-IDF Vectorizer ---
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    max_features=8000,
)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9,
    max_features=8000,
)

vectorizer = FeatureUnion([
    ("char", char_vectorizer),
    ("word", word_vectorizer),
])

Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

# --- 6️⃣ Classifier (calibrated logistic) ---
base_clf = LogisticRegression(
    C=2.0, solver="lbfgs", max_iter=3000, class_weight="balanced"
)
clf = CalibratedClassifierCV(base_clf, cv=3)
clf.fit(Xtr, y_train)

# --- 7️⃣ Evaluation ---
pred = clf.predict(Xte)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")

print("\n✅ Evaluation Results")
print(f"🎯 Accuracy: {acc:.4f}")
print(f"📊 Macro F1: {f1:.4f}\n")

report = classification_report(y_test, pred, output_dict=True)
print("📈 Per-intent metrics:")
for intent in unique_intents:
    if intent in report:
        prec = report[intent].get("precision", 0)
        rec = report[intent].get("recall", 0)
        f1i = report[intent].get("f1-score", 0)
        print(f"   {intent:<20} Prec={prec:.3f}  Rec={rec:.3f}  F1={f1i:.3f}")

print("\nDetailed report:")
print(classification_report(y_test, pred, digits=4))

cm = confusion_matrix(y_test, pred, labels=unique_intents)
cm_df = pd.DataFrame(cm, index=unique_intents, columns=unique_intents)
print("\n🔍 Confusion Matrix (partial view):")
print(cm_df.head())

# --- 8️⃣ Save artifacts ---
INTENT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(vectorizer, VEC_PATH)
joblib.dump(clf, CLF_PATH)

print(f"\n💾 Saved artifacts:")
print(f"   📁 {VEC_PATH.name}")
print(f"   📁 {CLF_PATH.name}")
print(f"🔁 SEED used: {SEED}")
print("🚀 Ready for FastAPI deployment.")
