# Cantilever-task1
A web-based sentiment analysis app that classifies movie reviews as Positive or Negative using a logistic regression model trained on the NLTK movie_reviews dataset.
# 🎬 Movie Review Sentiment Analyzer

This project is a web-based sentiment analysis system that predicts whether a movie review is **positive** or **negative**. It is built using:

- **Natural Language Processing (NLP)** techniques
- **Logistic Regression** classifier
- **NLTK's movie_reviews dataset**
- **Flask** for the web interface
- **HTML/CSS** for frontend design

---

## 🚀 Features

- Input any movie review text and get real-time sentiment prediction.
- Beautiful, responsive UI with styled output for sentiment.
- Model trained on real movie reviews using `TfidfVectorizer` + Logistic Regression.
- Lightweight and easy to deploy.

---

## 🧠 Model Details

- **Dataset:** NLTK `movie_reviews` (contains 1000 positive and 1000 negative reviews)
- **Vectorization:** TF-IDF (max features = 5000)
- **Algorithm:** Logistic Regression
- **Accuracy:** Evaluated using `classification_report` from `sklearn`.

---

## Folder Structure

├── app.py                 # Flask app
├── train_model.py         # Model training script
├── sentiment_model.pkl    # Saved model (TF-IDF + LR)
├── templates/
│   └── index.html         # Web interface

