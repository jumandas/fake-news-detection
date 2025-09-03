# app/predict.py
import pickle
from preprocessing import clean_text, load_vectorizer

def load_model(model_name="logistic_regression"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

def predict_news(text, model_name="logistic_regression"):
    model = load_model(model_name)
    vectorizer = load_vectorizer()
    text_clean = clean_text(text)
    text_vec = vectorizer.transform([text_clean])
    prediction = model.predict(text_vec)[0]
    return "REAL" if prediction == 1 else "FAKE"

if __name__ == "__main__":
    sample_text = "The president announced new policies today."
    print("Prediction:", predict_news(sample_text))
