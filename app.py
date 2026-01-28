from flask import Flask, render_template, request
import pandas as pd
from scipy.sparse import load_npz
import os
import urllib.request
import zipfile
import pickle

from src.recommender_cf import CollaborativeRecommender
from src.recommender_content import ContentRecommender
from src.recommender_hybrid import HybridRecommender


app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

MODEL_URL = (
    "https://huggingface.co/Sriraj01/"
    "electronics-recommender-artifact/resolve/main/model_artifacts.zip"
)
ARTIFACT_ZIP = "model_artifacts.zip"
REQUIRED_FILE = "models/als_model.pkl"

if not os.path.exists(REQUIRED_FILE):
    print("Downloading model artifacts...")
    urllib.request.urlretrieve(MODEL_URL, ARTIFACT_ZIP)

    if not zipfile.is_zipfile(ARTIFACT_ZIP):
        raise RuntimeError("Downloaded artifact is not a valid ZIP file")

    with zipfile.ZipFile(ARTIFACT_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Model artifacts downloaded successfully.")


cf = None
content = None
hybrid = None
interactions = None
df = None
asin_to_text = None
full_product_text = None


def load_artifacts():
    global cf, content, hybrid, interactions, df, asin_to_text, full_product_text

    if hybrid is not None:
        return

    print("Loading ML artifacts...")

    df = pd.read_csv("data/processed/electronics_subset.csv")
    df = df[["asin", "reviewText", "overall"]]

    asin_to_text = (
        df.dropna(subset=["reviewText"])
          .groupby("asin")["reviewText"]
          .first()
          .to_dict()
    )

    with open("models/product_text.pkl", "rb") as f:
        full_product_text = pickle.load(f)

    interactions = load_npz("data/processed/interactions.npz")

    cf = CollaborativeRecommender(
        "models/als_model.pkl",
        "models/user_mapping.pkl",
        "models/item_mapping.pkl"
    )

    content = ContentRecommender(
        "models/tfidf_vectorizer.pkl",
        "models/item_index.pkl",
        "models/product_text.pkl"
    )

    hybrid = HybridRecommender(cf, content)

    print("ML artifacts loaded successfully.")

def popular_items(df, top_n=5):
    return (
        df.groupby("asin")["overall"]
          .mean()
          .sort_values(ascending=False)
          .head(top_n)
          .index
          .tolist()
    )


def get_product_text(asin):
    if asin in asin_to_text:
        return asin_to_text[asin]
    if asin in full_product_text:
        return full_product_text[asin]
    return "No description available"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    load_artifacts()

    user_id = request.form.get("user_id")

    recommendations = hybrid.recommend(user_id, interactions, N=5)

    if not recommendations:
        recommendations = popular_items(df, top_n=5)
        message = "New user detected — showing popular items"
    else:
        message = "Personalized recommendations for you"

    display_recommendations = [
        f"{asin}: {get_product_text(asin)[:120]}..."
        for asin in recommendations
    ]

    return render_template(
        "results.html",
        user_id=user_id,
        recommendations=display_recommendations,
        message=message
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
