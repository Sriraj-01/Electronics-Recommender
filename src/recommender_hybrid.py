class HybridRecommender:
    def __init__(self, cf_model, content_model, alpha=0.7):
        self.cf = cf_model
        self.content = content_model
        self.alpha = alpha

    def recommend(self, user_id, user_interactions, N=10):
        cf_results = self.cf.recommend(user_id, user_interactions, N=20)

        if not cf_results:
            return []

        seed_asin = cf_results[0][0]
        content_boost = self.content.similar_items(seed_asin, top_n=20)

        scores = {}
        for asin, cf_score in cf_results:
            scores[asin] = self.alpha * cf_score

        for asin in content_boost:
            scores[asin] = scores.get(asin, 0) + (1 - self.alpha) * 0.5

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [asin for asin, _ in ranked[:N]]
