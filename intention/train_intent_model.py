# intention/train_intent_model.py
# ============================================
# 🧠 Train baseline intent model (TF-IDF + LogisticRegression)
# - Reads intention/intents.csv  (text,intent)
# - Trains & evaluates
# - Saves tfidf_vectorizer.pkl and intent_classifier.pkl
# ============================================

import os, csv, sys, json, joblib, random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]           # project root
INTENT_DIR = ROOT / "intention"
CSV_PATH = INTENT_DIR / "intents.csv"
VEC_PATH = INTENT_DIR / "tfidf_vectorizer.pkl"
CLF_PATH = INTENT_DIR / "intent_classifier.pkl"

SEED = 42
random.seed(SEED)

# --- 1) seed data if csv missing or basically empty ---
SEED_ROWS = [
    # check_balance
    ("Би дансныхаа үлдэгдлийг шалгамаар байна", "check_balance"),
    ("Үлдэгдэл хэд байгаа вэ", "check_balance"),
    ("Дансны баланс харуул", "check_balance"),
    ("Үлдэгдлээ хэлээд өгөөч", "check_balance"),
    ("Дансны үлдэгдэл сонирхож байна", "check_balance"),

    # show_transactions
    ("Гүйлгээний жагсаалт", "show_transactions"),
    ("Сүүлийн гүйлгээнүүдийг харуул", "show_transactions"),
    ("Өнөөдрийн гүйлгээг үзье", "show_transactions"),
    ("Дансны хуулга хэрэгтэй байна", "show_transactions"),
    ("Гүйлгээгээ хармаар байна", "show_transactions"),

    # lost_card
    ("Карт маань алга боллоо", "lost_card"),
    ("Картаа гээчихлээ", "lost_card"),
    ("Карт олдохгүй байна", "lost_card"),
    ("Карт блоклоорой", "lost_card"),
    ("Карт асуудалтай байна", "lost_card"),

    # transfer_money
    ("100 мянгыг найз руу шилжүүл", "transfer_money"),
    ("50000 төгрөг шилжүүлээрэй", "transfer_money"),
    ("Данс руу мөнгө явуул", "transfer_money"),
    ("Өөрийн данснаас шилжүүлэх", "transfer_money"),
    ("Амгалангийн данс руу 200 мянга", "transfer_money"),

    # open_account
    ("Шинэ данс нээх хүсэлтэй байна", "open_account"),
    ("Хадгаламжийн данс нээе", "open_account"),
    ("Данс нээхэд яах вэ", "open_account"),
    ("Хүүхдийн хадгаламж нээх", "open_account"),
    ("Шинэ данс нээмээр байна", "open_account"),

    # exchange_rate
    ("Валютын ханш хэд вэ", "exchange_rate"),
    ("Нэг доллар хэд вэ", "exchange_rate"),
    ("USD ханш хэлээд өг", "exchange_rate"),
    ("Евро ханш хэд байна", "exchange_rate"),
    ("Юанийн ханш", "exchange_rate"),

    # branch_hours
    ("Салбар хэдэн цагт ажилладаг вэ", "branch_hours"),
    ("Ажлын цаг хэд вэ", "branch_hours"),
    ("Амралтын өдөр нээх үү", "branch_hours"),
    ("Өнөөдөр хэдэд хаах вэ", "branch_hours"),
    ("Даваа-Баасан хэдээс хэд хүртэл", "branch_hours"),

    # contact_support
    ("Тусламж хэрэгтэй байна", "contact_support"),
    ("Оператортой холбож өг", "contact_support"),
    ("Харилцагчийн үйлчилгээтэй ярих", "contact_support"),
    ("Асуудал гарлаа, туслаач", "contact_support"),
    ("Дугаар нь хэд вэ тусламжийн", "contact_support"),

    # loan_info (doc intent)
    ("Танай банкны зээл, зээлийн хүү", "loan_info"),
    ("Зээл авахад бүрдүүлэх материал", "loan_info"),
    ("Ипотекийн зээлийн нөхцөл", "loan_info"),
    ("Хүний зээлийн хүү хэд вэ", "loan_info"),
    ("Зээлийн шугамын мэдээлэл", "loan_info"),

    # savings_info (doc intent)
    ("Хадгаламжийн хүү хэд вэ", "savings_info"),
    ("Хадгаламж нээхэд хамгийн бага дүн", "savings_info"),
    ("Хугацаат хадгаламжийн нөхцөл", "savings_info"),
    ("Хүүхдийн хадгаламжийн мэдээлэл", "savings_info"),
    ("Хадгаламжийн төрөл хэлээд өг", "savings_info"),

    # customer_info (doc intent)
    ("Дансны мэдээллээ шалгах", "customer_info"),
    ("Клиентийн мэдээлэлээ шинэчлэх", "customer_info"),
    ("Дансны төрөл, шимтгэлийн талаар", "customer_info"),
    ("Клиент бүртгэлийн тухай", "customer_info"),
    ("Данс холбох заавар", "customer_info"),
]

def ensure_csv():
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "intent"])
            w.writerows(SEED_ROWS)
        print(f"⚠️  {CSV_PATH} not found → created seed dataset with {len(SEED_ROWS)} rows.")
        return

    # if exists but has only header or very few rows → seed append
    df = pd.read_csv(CSV_PATH)
    if len(df) < 16:  # arbitrarily small
        with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(SEED_ROWS)
        print(f"⚠️  {CSV_PATH} had few rows → appended seed samples ({len(SEED_ROWS)}).")

# --- 2) load data ---
ensure_csv()
df = pd.read_csv(CSV_PATH).dropna()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Basic sanity
if not {"text","intent"}.issubset(df.columns):
    print("❌ intents.csv must have columns: text,intent")
    sys.exit(1)

print(f"📄 Dataset size: {len(df)}  |  Intents: {sorted(df['intent'].unique().tolist())}")

# --- 3) split ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["intent"], test_size=0.2, random_state=SEED, stratify=df["intent"]
)

# --- 4) vectorizer & model ---
# Word TF-IDF with uni/bi-grams works well for Mongolian; Unicode tokenization is OK.
vectorizer = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1, 2),
    lowercase=True,
    strip_accents=None
)

Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced")
clf.fit(Xtr, y_train)

# --- 5) eval ---
pred = clf.predict(Xte)
acc = accuracy_score(y_test, pred)
f1  = f1_score(y_test, pred, average="macro")
print("\n✅ Evaluation")

print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
print("\nPer-class report:")
print(classification_report(y_test, pred, digits=4))

# --- 6) save ---
INTENT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(vectorizer, VEC_PATH)
joblib.dump(clf, CLF_PATH)

print(f"\n💾 Saved: {VEC_PATH.name}, {CLF_PATH.name}")
print("👉 Your FastAPI router will now load these automatically.")
