import os
import sys
import pickle



BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from src.utils import clean_text


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
    raise FileNotFoundError(
        "spam_model.pkl not found. Run train_model.py first."
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "tfidf_vectorizer.pkl not found. Run train_model.py first."
    )


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)



def predict_sms(message):

    cleaned = clean_text(message)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction


def test_spam_message():

    message = "Congratulations! You have won ₹50000. Click here."

    prediction = predict_sms(message)

    assert prediction == 1


def test_ham_message():

    message = "Hello, are you coming to college today?"

    prediction = predict_sms(message)

    assert prediction == 0




if __name__ == "__main__":

    print("=" * 50)
    print("Running Tests")
    print("=" * 50)

    test_spam_message()

    print("✓ Spam Test Passed")

    test_ham_message()

    print("✓ Ham Test Passed")

    print("=" * 50)
    print("All Tests Passed Successfully")
    print("=" * 50)