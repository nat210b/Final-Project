from flask import Flask, render_template, request
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from pythainlp.util import maiyamok
import joblib
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
cvec_model = CountVectorizer(vocabulary=joblib.load('cvec2.pkl'))
logreg_model = joblib.load('Logis2.pkl')
stopword = "stopword.txt"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text_input']
        text = text.lower()
        text = word_tokenize(text,engine='newmm',keep_whitespace=False)
        text = [i for i in text if i not in stopword]
        text = " ".join(text)
        tokens=text
        features = cvec_model.transform(pd.Series([tokens]))
        # features = cvec_model.transform([' '.join(tokens)]).toarray()
        prediction = logreg_model.predict(features)
        if( prediction == "Positive"):sentiment="บวก"
        else: sentiment="ลบ"
    return render_template('index.html', sentiment= sentiment,sentence=text)
if __name__ == '__main__':
    app.run(debug=True)
    from flask import Flask, render_template