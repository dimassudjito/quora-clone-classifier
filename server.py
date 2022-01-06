# For API
from flask import Flask, request, jsonify
import traceback
import joblib
# For classifier model
import pandas as pd
import numpy as np
# For text preprocessing
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            input = request.json["msg"]
            prediction = make_prediction(input)
            return jsonify({'prediction': prediction})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return jsonify({'trace': "Model not found"})

def make_prediction(msg):
    input = preprocess_text(msg)
    print(input)
    prediction = classifier.predict(input)
    print(prediction)
    return True if prediction == [1] else False

def preprocess_text(msg):
    # Remove punctuation and lowercase the sentence
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower()
    review = review.split()
    # Stem the sentence
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # stem non-stopwords words
    review = ' '.join(review)
    # Remove non-frequent words
    review = cv.transform([review]).toarray()

    return review

if __name__ == '__main__':
    classifier = joblib.load("model.pkl")
    cv = joblib.load("count_vectorizer.pkl")
    print ('Model loaded')
    app.run(port=5050, debug=True)