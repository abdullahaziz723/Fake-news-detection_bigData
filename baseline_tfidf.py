# baseline_tfidf.py
import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Paths
DATA_PATH = os.path.join("dataset", "combined_news.csv")
MODEL_OUT = "tfidf_logreg_model.joblib"

# Load
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
df = df.dropna(subset=['title','text','label']).reset_index(drop=True)

# Combine title + text
df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

# Clean text function
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    tokens = [w for w in s.split() if w not in STOPWORDS]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

print("Cleaning texts (this can take a while)...")
df['clean'] = df['content'].apply(clean_text)

# Labels: ensure numeric 0/1
y = df['label'].astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(df['clean'], y, test_size=0.2, random_state=42, stratify=y)

# Vectorizer and model pipeline (we'll run manually)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Choose classifier: LogisticRegression (fast + interpretable)
clf = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)

print("Training classifier...")
clf.fit(X_train_tfidf, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test_tfidf)
y_proba = clf.predict_proba(X_test_tfidf)[:,1] if hasattr(clf, "predict_proba") else None

print("Accuracy:", accuracy_score(y_test, y_pred))
if y_proba is not None:
    try:
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        pass
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.show()

# Save model and vectorizer
joblib.dump({'tfidf': tfidf, 'clf': clf}, MODEL_OUT)
print("Saved model to", MODEL_OUT)

# Example predict function
def predict_text(text):
    c = clean_text(text)
    v = tfidf.transform([c])
    pred = clf.predict(v)[0]
    prob = clf.predict_proba(v)[0][1] if hasattr(clf, "predict_proba") else None
    return pred, prob

# Example usage
example = "NASA announces water on Mars and new rover mission"
print("Example prediction:", predict_text(example))
