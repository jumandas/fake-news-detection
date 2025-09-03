# app/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

from preprocessing import clean_text, create_vectorizer

# Load dataset
df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

df_fake["label"] = 0  # fake
df_true["label"] = 1  # real
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Clean text
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
os.makedirs("models", exist_ok=True)
vectorizer, X_train_vec = create_vectorizer(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
print("Naive Bayes:\n", classification_report(y_test, y_pred_nb))

with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(nb, f)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)
y_pred_lr = lr.predict(X_test_vec)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))

with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr, f)

print("✅ Models trained and saved in /models")
