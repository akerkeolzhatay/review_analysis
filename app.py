import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, redirect, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import re

app = Flask(__name__)
app.secret_key = "super_secret_key_123"     

app.config["MONGO_URI"] = "mongodb+srv://ecommerce:finalml@qazsports.7ffyl6z.mongodb.net/review_app?retryWrites=true&w=majority"
mongo = PyMongo(app)

binary_pack = joblib.load("binary_model_and_vectorizer.joblib")
binary_model = binary_pack["model"]
binary_vectorizer = binary_pack["vectorizer"]

multi_pack = joblib.load("multiclass_model_and_vectorizer.joblib")
multi_model = multi_pack["model"]
multi_vectorizer = multi_pack["vectorizer"]

ml_pack = joblib.load("multilabel_model_and_tools.joblib")
ml_model = ml_pack["model"]
ml_vectorizer = ml_pack["vectorizer"]
ml_mlb = ml_pack["mlb"]

# --- NLP clean function ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def predict_binary(text):
    cleaned = clean_text(text)
    vec = binary_vectorizer.transform([cleaned])
    proba = binary_model.predict_proba(vec)[0][1]
    label = "Positive" if proba >= 0.5 else "Negative"
    return label, float(proba)

def predict_multiclass(text):
    cleaned = clean_text(text)
    vec = multi_vectorizer.transform([cleaned])
    probs = multi_model.predict_proba(vec)[0]
    star = int(np.argmax(probs)) + 1
    return star, probs

def predict_multilabel(text, top_n=3):
    cleaned = clean_text(text)
    vec = ml_vectorizer.transform([cleaned])
    probs_list = ml_model.predict_proba(vec)

    arr = []
    for p in probs_list:
        p = np.array(p)
        if p.ndim == 2:
            arr.append(p[:, 1])
        else:
            arr.append(p)
    arr = np.array(arr).flatten()

    top_indices = np.argsort(arr)[-top_n:]  
    labels = [ml_mlb.classes_[i] for i in top_indices]
    return labels, arr.tolist()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if mongo.db.users.find_one({"username": username}):
            return "User already exists!"

        mongo.db.users.insert_one({
            "username": username,
            "password": generate_password_hash(password)
        })
        return redirect("/login")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = mongo.db.users.find_one({"username": username})
        if not user or not check_password_hash(user["password"], password):
            return "Invalid username or password!"

        session["user"] = username
        return redirect("/")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# --- MAIN PAGE ---
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        review = request.form.get("review")

        # predictions
        b_label, b_proba = predict_binary(review)
        m_star, m_probs = predict_multiclass(review)
        ml_labels, ml_probs = predict_multilabel(review)

        # save to MongoDB
        mongo.db.reviews.insert_one({
            "user": session["user"],
            "review": review,
            "binary_label": b_label,
            "binary_proba": b_proba,
            "multiclass_stars": m_star,
            "multiclass_probs": m_probs.tolist(),
            "multilabel_labels": ml_labels,
            "multilabel_probs": ml_probs
        })

        return render_template(
            "index.html",
            review=review,
            b_label=b_label,
            b_proba=b_proba,
            m_star=m_star,
            m_probs=m_probs,
            ml_labels=ml_labels,
            ml_probs=ml_probs,  
            ml_classes=list(ml_mlb.classes_)
        )



    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
