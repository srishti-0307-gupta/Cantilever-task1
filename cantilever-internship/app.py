from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("sentiment_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = model.predict([review])[0]

    # Map 'pos' → 'positive', 'neg' → 'negative'
    if prediction == 'pos':
        sentiment = 'positive'
    elif prediction == 'neg':
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
