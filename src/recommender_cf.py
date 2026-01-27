import pickle
import numpy as np

class CollaborativeRecommender:
    def __init__(self, model_path, user_map_path, item_map_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(user_map_path, "rb") as f:
            self.user_encoder = pickle.load(f)
        with open(item_map_path, "rb") as f:
            self.item_encoder = pickle.load(f)

    def recommend(self, user_raw_id, user_interactions, N=10):
        if user_raw_id not in self.user_encoder.classes_:
            return []

        user_id = self.user_encoder.transform([user_raw_id])[0]

        item_ids, scores = self.model.recommend(
            user_id,
            user_interactions[user_id],
            N=N
        )

        asins = self.item_encoder.inverse_transform(item_ids)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return list(zip(asins, scores))
