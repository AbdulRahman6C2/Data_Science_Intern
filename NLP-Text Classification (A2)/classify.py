import pandas as pd
import nltk
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

nltk.download("stopwords")
nltk.download("wordnet")

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("Sentiment_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns)

# Rename columns to standard names
df = df.rename(columns={
    "Text": "text",
    "Sentiment": "label"
})

# Keep only required columns
df = df[['text', 'label']].dropna()

print("\nSample data:")
print(df.head())

# ===============================
# 2. Label Distribution
# ===============================
print("\nLabel distribution:")
print(df['label'].value_counts())

X = df['text']
y = df['label']

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nUnique labels in y_train:", y_train.unique())
print("Unique labels in y_test:", y_test.unique())

# ===============================
# 4. Vectorization
# ===============================
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 5. Model Training
# ===============================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ===============================
# 6. Prediction
# ===============================
y_pred = model.predict(X_test_vec)
print("\nUnique labels in y_pred:", set(y_pred))

# ===============================
# 7. Evaluation (Safe)
# ===============================
labels = sorted(y.unique())

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)
