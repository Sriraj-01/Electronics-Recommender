import pickle
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, tfidf_path, item_index_path, product_text_path):
        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(item_index_path, "rb") as f:
            self.item_index = pickle.load(f)
        with open(product_text_path, "rb") as f:
            self.product_text = pickle.load(f)

        self.tfidf_matrix = self.tfidf.transform(
            self.product_text["clean_text"]
        )

    def similar_items(self, asin, top_n=10):
        if asin not in self.item_index:
            return []

        idx = self.item_index[asin]
        sims = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        )[0]

        top_idx = sims.argsort()[::-1][1:top_n+1]
        return self.product_text.iloc[top_idx]["asin"].tolist()
