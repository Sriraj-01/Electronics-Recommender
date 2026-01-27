from flask import Flask, jsonify, render_template, request
import pandas as pd
from scipy.sparse import load_npz
import pickle
from src.recommender_cf import CollaborativeRecommender
from src.recommender_content import ContentRecommender
from src.recommender_hybrid import HybridRecommender

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

df = pd.read_csv("data/processed/electronics_subset.csv")
df = df[
    ["asin", "reviewText", "overall"]
]
asin_to_Text = (
    df.dropna(subset=["reviewText"])
    .groupby("asin")["reviewText"]
    .first()
    .to_dict()
)

interactions = load_npz("data/processed/interactions.npz")

def popular_items(df, top_n = 5):
    return (
        df.groupby("asin")["overall"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

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
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")
    recommendations = hybrid.recommend(user_id, interactions, N=5)
    if not recommendations:
        recommendations = popular_items(df, top_n=5)
        message = "New User! Here are some popular items"
    else:
        message = "Presonalized recommendations for you"
    display_recommendations = [
        f"{asin}: {asin_to_Text.get(asin, 'No reviewText available')[:120]}..."
        for asin in recommendations
    ]
    return render_template("results.html", user_id=user_id, recommendations=display_recommendations, message=message)

if __name__ == "__main__":
    app.run(debug=True)

