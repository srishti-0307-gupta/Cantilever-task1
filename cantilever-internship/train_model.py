import nltk
import random
import string
import pickle
import pandas as pd
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 📥 Download required NLTK resources
nltk.download('movie_reviews')
nltk.download('stopwords')

# 📊 Load dataset from nltk
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# 🔀 Shuffle the data
random.shuffle(documents)

# 🔡 Preprocess text
def preprocess(doc):
    stop_words = set(stopwords.words('english'))
    return ' '.join([
        word.lower() for word in doc
        if word.lower() not in stop_words and word not in string.punctuation
    ])

# 🧹 Apply preprocessing
texts = [preprocess(words) for words, label in documents]
labels = [label for words, label in documents]

# 🧠 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# ⚙️ Create a pipeline (TF-IDF + Logistic Regression)
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

# 🏋️ Train the model
model_pipeline.fit(X_train, y_train)

# 🧪 Evaluate
y_pred = model_pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 💾 Save the model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("\n✅ Model training complete. File saved as sentiment_model.pkl")
