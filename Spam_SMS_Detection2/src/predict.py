import os
import pickle

from utils import clean_text

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "spam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)



if not os.path.exists(MODEL_PATH):

    print("=" * 60)
    print("ERROR : spam_model.pkl not found.")
    print("Run train_model.py first.")
    print("=" * 60)
    exit()

if not os.path.exists(VECTORIZER_PATH):

    print("=" * 60)
    print("ERROR : tfidf_vectorizer.pkl not found.")
    print("Run train_model.py first.")
    print("=" * 60)
    exit()



print("=" * 60)
print("Loading Machine Learning Model...")
print("=" * 60)

with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:

    vectorizer = pickle.load(file)

print("Model Loaded Successfully.")


def predict_sms(message):

    cleaned_message = clean_text(message)

    vector = vectorizer.transform([cleaned_message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)

    spam_probability = probability[0][1] * 100
    ham_probability = probability[0][0] * 100

    return prediction, spam_probability, ham_probability




print("\nSpam SMS Detection System")
print("=" * 60)
print("Type your SMS below.")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    sms = input("\nEnter SMS : ")

    if sms.strip().lower() == "exit":

        print("\nProgram Closed Successfully.")
        break

    if sms.strip() == "":

        print("Please enter a valid message.")
        continue

    prediction, spam_prob, ham_prob = predict_sms(sms)

    print("\nPrediction Result")
    print("-" * 40)

    if prediction == 1:

        print("🚨 SPAM MESSAGE")

    else:

        print("✅ HAM (SAFE) MESSAGE")

    print(f"Spam Probability : {spam_prob:.2f}%")
    print(f"Ham Probability  : {ham_prob:.2f}%")

    print("-" * 40)