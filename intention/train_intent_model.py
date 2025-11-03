# ============================================
# 🧠 train_intent_model.py (v3.0 — Final Intent CSV Edition)
# ✅ Uses clean intent.csv (no seeding required)
# ✅ Normalizes labels + ensures stable SEED accuracy
# ✅ Prints per-intent accuracy and macro metrics
# ============================================

import os, sys, joblib, random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- 1️⃣ Paths ---
ROOT = Path(__file__).resolve().parents[1]
INTENT_DIR = ROOT / "intention"
CSV_PATH = INTENT_DIR / "intents.csv"
VEC_PATH = INTENT_DIR / "tfidf_vectorizer.pkl"
CLF_PATH = INTENT_DIR / "intent_classifier.pkl"

SEED = 42
random.seed(SEED)

# --- 2️⃣ Load & verify CSV ---
if not CSV_PATH.exists():
    print(f"❌ Missing {CSV_PATH}. Please place your final intents.csv there.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH).dropna()

if not {"text", "intent"}.issubset(df.columns):
    print("❌ intents.csv must have columns: text,intent")
    sys.exit(1)

# Normalize labels (lowercase, trim)
df["intent"] = df["intent"].astype(str).str.strip().str.lower()

# Shuffle for randomness but keep deterministic seed
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

unique_intents = sorted(df["intent"].unique().tolist())
print(f"📄 Dataset size: {len(df)}  |  Intents: {len(unique_intents)}")
print(f"🧩 Intents: {unique_intents}")

# --- 3️⃣ Train/Test split (safe stratify) ---
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=SEED, stratify=df["intent"]
    )
except ValueError:
    print("⚠️ Some intents too rare for stratify → using random split instead.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=SEED
    )

# --- 4️⃣ Vectorizer & Model ---
vectorizer = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1, 2),
    lowercase=True,
    strip_accents=None
)

Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, y_train)

# --- 5️⃣ Evaluation ---
pred = clf.predict(Xte)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")

print("\n✅ Evaluation Results")
print(f"🎯 Accuracy: {acc:.4f}")
print(f"📊 Macro F1: {f1:.4f}\n")

# Per-intent accuracy
print("📈 Per-intent performance:")
report = classification_report(y_test, pred, output_dict=True)
for intent in unique_intents:
    if intent in report:
        print(f"   {intent:<20}  Acc={report[intent]['precision']:.3f}  F1={report[intent]['f1-score']:.3f}")

print("\nDetailed report:")
print(classification_report(y_test, pred, digits=4))

# --- 6️⃣ Confusion Matrix (optional quick view) ---
cm = confusion_matrix(y_test, pred, labels=unique_intents)
cm_df = pd.DataFrame(cm, index=unique_intents, columns=unique_intents)
print("\n🔍 Confusion Matrix (partial view):")
print(cm_df.head())

# --- 7️⃣ Save artifacts ---
INTENT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(vectorizer, VEC_PATH)
joblib.dump(clf, CLF_PATH)

print(f"\n💾 Saved model and vectorizer:")
print(f"   📁 {VEC_PATH.name}")
print(f"   📁 {CLF_PATH.name}")
print(f"\n🔁 SEED used: {SEED}")
print("👉 Your FastAPI router will now load these automatically.")
