import os
import pickle
import sys

from flask import Flask, render_template, request, jsonify


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import clean_text

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")


if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    model = None
    vectorizer = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model files nahi mili! Pehle model train karein."}), 500
        
    data = request.form.get("message")
    if not data:
        return render_template("index.html", prediction="Kripya koi message likhein.")
        
   
    cleaned_data = clean_text(data)
    vectorized_data = vectorizer.transform([cleaned_data])
    
   
    prediction_code = model.predict(vectorized_data)[0]
    result = "SPAM" if prediction_code == 1 else "HAM (Safe)"
    
    return render_template("index.html", prediction=result, original_text=data)

if __name__ == "__main__":
    app.run(debug=True)