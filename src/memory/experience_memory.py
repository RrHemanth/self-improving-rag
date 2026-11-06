import json
from pathlib import Path

import numpy as np


class ExperienceMemory:
    def __init__(
        self,
        path="data/memory/experiences.json"
    ):
        self.path = Path(path)
        self.experiences = []

        self._past_embeddings = None
        self._cached_queries = []
        self._embedding_model = None

        if self.path.exists():
            with open(self.path) as file:
                self.experiences = json.load(file)

    def add(self, experience):
        self.experiences.append(experience)
        self.save()

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(self.path, "w") as file:
            json.dump(
                self.experiences,
                file,
                indent=2
            )

    def _get_past_embeddings(
        self,
        past_queries,
        embedding_model
    ):
        model_changed = (
            self._embedding_model is not embedding_model
        )

        queries_changed = (
            past_queries[:len(self._cached_queries)]
            != self._cached_queries
        )

        if (
            self._past_embeddings is None
            or model_changed
            or queries_changed
        ):
            self._past_embeddings = embedding_model.encode(
                past_queries
            )

            self._cached_queries = past_queries.copy()
            self._embedding_model = embedding_model

        elif len(past_queries) > len(self._cached_queries):
            new_queries = past_queries[
                len(self._cached_queries):
            ]

            new_embeddings = embedding_model.encode(
                new_queries
            )

            self._past_embeddings = np.vstack([
                self._past_embeddings,
                new_embeddings
            ])

            self._cached_queries.extend(new_queries)

        return self._past_embeddings

    def suggest_strategy(
        self,
        query,
        embedding_model,
        strategies,
        default_strategy,
        top_k=5,
        min_similarity=0.6
    ):
        if not self.experiences:
            return default_strategy, {
                "has_memory_match": False,
                "neighbors": [],
                "scores": {}
            }

        past_queries = [
            experience["query"]
            for experience in self.experiences
        ]

        past_embeddings = self._get_past_embeddings(
            past_queries,
            embedding_model
        )

        query_embedding = embedding_model.encode(
            [query]
        )[0]

        similarities = np.dot(
            past_embeddings,
            query_embedding
        )

        indices = similarities.argsort()[::-1][
            :top_k
        ]

        totals = {
            strategy: 0.0
            for strategy in strategies
        }

        weights = {
            strategy: 0.0
            for strategy in strategies
        }

        neighbors = []

        for index in indices:
            similarity = float(
                similarities[index]
            )

            if similarity < min_similarity:
                continue

            experience = self.experiences[index]

            neighbors.append({
                "query": experience["query"],
                "similarity": similarity,
                "best_strategy": experience[
                    "best_strategy"
                ]
            })

            for attempt in experience["attempts"]:
                strategy = attempt["strategy"]

                if strategy not in totals:
                    continue

                score = attempt["overall_score"]

                if attempt["success"]:
                    utility = score
                else:
                    utility = score - 1.0

                totals[strategy] += (
                    similarity * utility
                )

                weights[strategy] += similarity

        if not neighbors:
            return default_strategy, {
                "has_memory_match": False,
                "neighbors": [],
                "scores": {}
            }

        scores = {}

        for strategy in strategies:
            if weights[strategy] > 0:
                scores[strategy] = (
                    totals[strategy]
                    / weights[strategy]
                )

        if not scores:
            return default_strategy, {
                "has_memory_match": False,
                "neighbors": neighbors,
                "scores": {}
            }

        best_strategy = max(
            scores,
            key=scores.get
        )

        return best_strategy, {
            "has_memory_match": True,
            "neighbors": neighbors,
            "scores": scores
        }