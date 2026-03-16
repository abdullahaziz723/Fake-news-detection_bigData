# 📰 Fake News Detection using Big Data & NLP

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![PySpark](https://img.shields.io/badge/Apache_Spark-PySpark-red?style=flat&logo=apachespark)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=flat)
![Accuracy](https://img.shields.io/badge/Accuracy-98.7%25-brightgreen?style=flat)

A machine learning system that detects fake news articles using **Natural Language Processing (NLP)** and **Big Data** technologies. Built with two implementations — a fast scikit-learn baseline and a scalable Apache Spark pipeline.

---

## 🎯 Problem Statement

Misinformation and fake news spread rapidly on social media platforms, causing real-world harm. This project automates the detection of fake vs. real news articles using ML, helping flag unreliable content at scale.

---

## 📊 Results

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|---------|
| Baseline (TF-IDF + Logistic Regression) | **98.7%** | **0.99** | — |
| Big Data (PySpark Pipeline) | — | **0.984** | **0.978** |

---

## 🗂️ Dataset

- **Source:** [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files:** `Fake.csv` + `True.csv` → merged into `combined_news.csv`
- **Size:** ~44,819 news articles
- **Labels:** `0` = Fake News, `1` = Real News

---

## 🏗️ Project Structure
```
Fake-news-detection_bigData/
│
├── baseline_tfidf.py          # Scikit-learn baseline model
├── prepare_dataset.py         # Data loading and preprocessing
├── spark_pipeline.py          # PySpark big data pipeline
├── tfidf_logreg_model.joblib  # Saved trained model
├── requirements.txt           # Python dependencies
├── spark_fake_news_model/     # Saved Spark model artifacts
└── README.md
```

---

## ⚙️ Methodology

### Step 1 — Data Preprocessing
- Combined Fake.csv and True.csv with labels
- Removed punctuation, URLs, and stopwords
- Applied text normalization and stemming (NLTK)

### Step 2 — Feature Extraction
- **TF-IDF Vectorization** — converts text into numerical feature vectors based on word importance

### Step 3 — Model Training

**Baseline Model** (scikit-learn):
```
Text → TF-IDF Vectorizer → Logistic Regression → Prediction
```

**Big Data Model** (PySpark):
```
Text → Tokenizer → StopWordsRemover → HashingTF → IDF → Logistic Regression → Prediction
```

### Step 4 — Evaluation
- Accuracy, F1-Score, ROC-AUC computed on held-out test set
- Confusion matrix visualized with Seaborn/Matplotlib

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/abdullahaziz723/Fake-news-detection_bigData.git
cd Fake-news-detection_bigData
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
```bash
python prepare_dataset.py
```

### 4. Run the baseline model
```bash
python baseline_tfidf.py
```

### 5. Run the PySpark pipeline (requires Apache Spark)
```bash
python spark_pipeline.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML Library | Scikit-learn |
| Big Data | Apache Spark (PySpark) |
| NLP | NLTK, TF-IDF |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | VS Code / Jupyter Notebook |

---

## 💡 Key Concepts Demonstrated

- **NLP Pipeline** — text cleaning, tokenization, vectorization
- **TF-IDF** — Term Frequency–Inverse Document Frequency for feature extraction
- **Logistic Regression** — binary classification (Fake vs Real)
- **Apache Spark** — distributed processing for large-scale datasets
- **Model Persistence** — saving and loading trained models with joblib
- **Evaluation Metrics** — Accuracy, Precision, Recall, F1, ROC-AUC

---

## 🔮 Future Scope

- Real-time detection via social media APIs (Twitter/X, Reddit)
- Deep learning models — LSTMs or BERT Transformers for better accuracy
- Web app deployment using Flask or FastAPI
- Cloud deployment on AWS/GCP for production scale

---

## 👤 Author

**Abdullah Aziz**  
GitHub: [@abdullahaziz723](https://github.com/abdullahaziz723)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
