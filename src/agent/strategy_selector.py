import numpy as np


class StrategySelector:
    def __init__(self, experiences, embedding_model):
        self.experiences = experiences
        self.embedding_model = embedding_model

        queries = [
            experience["query"]
            for experience in experiences
        ]

        self.query_embeddings = embedding_model.encode(queries)

    def select(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])[0]

        similarities = np.dot(
            self.query_embeddings,
            query_embedding
        )

        top_indices = similarities.argsort()[::-1][:top_k]

        votes = {}
        neighbors = []

        for index in top_indices:
            experience = self.experiences[index]
            strategy = experience["best_strategy"]
            similarity = float(similarities[index])

            votes[strategy] = votes.get(strategy, 0) + max(similarity, 0)

            neighbors.append({
                "query": experience["query"],
                "strategy": strategy,
                "similarity": similarity
            })

        best_strategy = max(
            votes,
            key=votes.get
        )

        return best_strategy, {
            "neighbors": neighbors,
            "votes": votes
        }

    def add_experience(self, experience):
        for i, item in enumerate(self.experiences):
            if item["query"] == experience["query"]:
                self.experiences[i] = experience
                return

        self.experiences.append(experience)

        embedding = self.embedding_model.encode(
            [experience["query"]]
        )

        self.query_embeddings = np.vstack([
            self.query_embeddings,
            embedding
        ])